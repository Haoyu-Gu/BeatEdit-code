#!/usr/bin/env python3
"""MusicXML -> piano NPZ converter (the format consumed by ``PianoDataset``).

Produces one ``.npz`` per score, laid out as::

    measure_{i}: uint8 (C, 88, T) per measure
        C = 4 by default:  [treble_sus, treble_ons, bass_sus, bass_ons]
        C = 6 with --velocity:
                           [treble_sus, treble_ons, treble_vel,
                            bass_sus,   bass_ons,   bass_vel]
        T = TICKS_PER_BEAT x quarter-lengths of the measure
            (TICKS_PER_BEAT = 4, i.e. the sixteenth-note grid, tau = 4 in the paper;
             a 4/4 measure is therefore 16 steps)
        pitch axis: index 0 = MIDI 21 (A0), index 87 = MIDI 108 (C8), ascending
        sus/ons channels are binary {0, 1}; vel holds MIDI velocity 0-127.
    metadata: pickled dict, see ``_build_metadata``.

The sustain/onset channels are what the ternary Beat Encoding is built from
(0 = silent, 1 = onset, 2 = sustain continuation; see the paper, Appendix A).
The velocity channels are optional and are ignored by the default single-track
pipeline -- ``PianoRollTokenizer`` reads them only when constructed with
``use_velocities=True``.

Usage
-----
    # single file
    python data_prep/xml2npz.py score.mxl --output-dir data/npz

    # whole directory, 8 worker processes
    python data_prep/xml2npz.py data/musescore/ --output-dir data/npz --workers 8

    # keep per-note velocity (6-channel output)
    python data_prep/xml2npz.py data/musescore/ --output-dir data/npz --velocity

Adapted from the BEAT reference implementation
(https://github.com/Lekai-Qian/BEAT-code, ``data_prep/xml2pianonpz.py``),
retargeted to the BeatEdit NPZ layout: 4 channels, tau = 4 steps per beat.
"""

import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import music21 as m21
except ImportError:  # pragma: no cover
    sys.exit("music21 is required: pip install music21")

warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# Constants (must stay in sync with src/encoding/scheme_*/config.py)
# ---------------------------------------------------------------------------

TICKS_PER_BEAT = 4          # tau: sixteenth-note grid, 4 steps per quarter note
PITCH_RANGE = 88            # 88 piano keys
MIN_PITCH = 21              # MIDI 21 = A0
MAX_PITCH = 108             # MIDI 108 = C8
DEFAULT_VELOCITY = 64

# Time signatures the token vocabulary can represent; the index is the
# TIME_SIG token offset used by the encoders (see token2midi.time_sig_map).
TIME_SIGNATURE_IDX = {'4/4': 0, '3/4': 1, '2/4': 2, '6/8': 3, '2/2': 4}

MIN_MEASURES = 16           # skip fragments
MAX_MEASURES = 300          # skip outliers


def _first_time_signature(score) -> Optional[Any]:
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            if measure.timeSignature is not None:
                return measure.timeSignature
    return None


def _tempo_info(score) -> Tuple[Optional[float], Optional[str]]:
    for tempo in score.flatten().getElementsByClass(m21.tempo.TempoIndication):
        bpm = getattr(tempo, 'number', None)
        text = getattr(tempo, 'text', None) or getattr(tempo, 'name', None)
        if bpm:
            return float(bpm), (str(text) if text else None)
        if text:
            return None, str(text)
    return None, None


def _key_info(score) -> Tuple[Optional[str], int]:
    """Return (key name, number of sharps); flats are negative."""
    for element in score.flatten().getElementsByClass(m21.key.KeySignature):
        sharps = int(getattr(element, 'sharps', 0) or 0)
        if isinstance(element, m21.key.Key):
            return str(element), sharps
        return None, sharps
    return None, 0


def _measure_boundaries(ref_part, ticks_per_beat: int) -> List[Dict[str, Any]]:
    """Tick span of every measure of the reference part, in score order."""
    out, cursor = [], 0
    for idx, measure in enumerate(ref_part.getElementsByClass('Measure')):
        dur = max(1, int(round(measure.duration.quarterLength * ticks_per_beat)))
        out.append({'index': idx, 'start': cursor, 'end': cursor + dur, 'duration': dur})
        cursor += dur
    return out


def _get_velocity(element) -> int:
    """MIDI velocity (1-127) of a Note/Chord, falling back to a default."""
    try:
        vol = element.volume
        if vol is not None:
            if vol.velocity is not None:
                return int(np.clip(vol.velocity, 1, 127))
            realized = vol.getRealized()
            if realized:
                return int(np.clip(round(realized * 127), 1, 127))
    except Exception:
        pass
    return DEFAULT_VELOCITY


def _extract_part_notes(part, boundaries, ticks_per_beat: int) -> List[Dict[str, Any]]:
    """Flatten one part into {pitch, abs_start, duration, is_tie_continuation, velocity}.

    Notes tied from a previous note do not get an onset: a tied note is a single
    sustained event, which is what the ternary pattern encodes as a continuation.
    """
    notes: List[Dict[str, Any]] = []
    start_by_idx = {mb['index']: mb['start'] for mb in boundaries}
    tied_pitches = set()

    for m_idx, measure in enumerate(part.getElementsByClass('Measure')):
        m_start = start_by_idx.get(m_idx)
        if m_start is None:
            continue
        for element in measure.flatten().notes:
            if not isinstance(element, (m21.note.Note, m21.chord.Chord)):
                continue
            abs_start = m_start + int(round(element.offset * ticks_per_beat))
            duration = max(1, int(round(element.duration.quarterLength * ticks_per_beat)))
            velocity = _get_velocity(element)

            if isinstance(element, m21.note.Note):
                pitches, ties = [element.pitch.midi], [element.tie]
            else:
                pitches = [p.midi for p in element.pitches]
                ties = [getattr(element, 'tie', None)] * len(pitches)

            for pitch, tie in zip(pitches, ties):
                if not (MIN_PITCH <= pitch <= MAX_PITCH):
                    continue
                is_continuation = pitch in tied_pitches
                if tie is not None:
                    if tie.type == 'start':
                        tied_pitches.add(pitch)
                    elif tie.type == 'stop':
                        tied_pitches.discard(pitch)
                elif is_continuation:
                    tied_pitches.discard(pitch)
                    is_continuation = False
                notes.append({
                    'pitch': pitch,
                    'abs_start': abs_start,
                    'duration': duration,
                    'is_tie_continuation': is_continuation,
                    'velocity': velocity,
                })
    return notes


def _notes_to_rolls(notes, total_length: int):
    """Rasterize notes into (sustain, onset, velocity) piano rolls."""
    sustain = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    onset = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    velocity = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    for note in notes:
        start = note['abs_start']
        if start >= total_length:
            continue
        p = note['pitch'] - MIN_PITCH
        end = min(start + note['duration'], total_length)
        sustain[p, start:end] = 1
        if not note['is_tie_continuation']:
            onset[p, start] = 1
        velocity[p, start:end] = note['velocity']
    return sustain, onset, velocity


class XMLToNPZ:
    """Convert a two-staff piano MusicXML into the BeatEdit NPZ layout."""

    def __init__(self, ticks_per_beat: int = TICKS_PER_BEAT,
                 with_velocity: bool = False,
                 min_measures: int = MIN_MEASURES,
                 max_measures: int = MAX_MEASURES):
        self.ticks_per_beat = ticks_per_beat
        self.with_velocity = with_velocity
        self.min_measures = min_measures
        self.max_measures = max_measures

    def convert(self, xml_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        score = m21.converter.parse(xml_path)
        parts = list(score.parts)
        if len(parts) < 2:
            raise ValueError(f"need >=2 staves (treble + bass), got {len(parts)}")

        n_measures = len(parts[0].getElementsByClass('Measure'))
        if not (self.min_measures <= n_measures <= self.max_measures):
            raise ValueError(
                f"measure count {n_measures} outside [{self.min_measures}, {self.max_measures}]")

        ts_obj = _first_time_signature(score) or m21.meter.TimeSignature('4/4')
        ts_string = ts_obj.ratioString
        if ts_string not in TIME_SIGNATURE_IDX:
            raise ValueError(
                f"time signature {ts_string} not representable "
                f"(supported: {sorted(TIME_SIGNATURE_IDX)})")

        boundaries = _measure_boundaries(parts[0], self.ticks_per_beat)
        if not boundaries:
            raise ValueError("no measures after boundary extraction")
        total_length = boundaries[-1]['end']

        treble = _notes_to_rolls(
            _extract_part_notes(parts[0], boundaries, self.ticks_per_beat), total_length)
        bass = _notes_to_rolls(
            _extract_part_notes(parts[1], boundaries, self.ticks_per_beat), total_length)

        # Steps per measure implied by the piece's time signature. Measures that
        # do not match it (pickup bars, irregular bars) cannot be tokenized on
        # the beat grid and are dropped; the kept indices go into valid_measures.
        steps_per_measure = int(round(ts_obj.barDuration.quarterLength * self.ticks_per_beat))

        segments: List[np.ndarray] = []
        valid_measures: List[int] = []
        for mb in boundaries:
            if mb['duration'] != steps_per_measure:
                continue
            s, e = mb['start'], mb['end']
            if self.with_velocity:
                seg = np.stack([treble[0][:, s:e], treble[1][:, s:e], treble[2][:, s:e],
                                bass[0][:, s:e], bass[1][:, s:e], bass[2][:, s:e]], axis=0)
            else:
                seg = np.stack([treble[0][:, s:e], treble[1][:, s:e],
                                bass[0][:, s:e], bass[1][:, s:e]], axis=0)
            segments.append(seg.astype(np.uint8))
            valid_measures.append(mb['index'])

        if not segments:
            raise ValueError("no measures matched the time signature grid")

        metadata = self._build_metadata(
            score, ts_string, steps_per_measure, total_length,
            len(boundaries), valid_measures)
        return segments, metadata

    def _build_metadata(self, score, ts_string, steps_per_measure, total_length,
                        original_measures, valid_measures) -> Dict[str, Any]:
        bpm, tempo_text = _tempo_info(score)
        key_name, key_idx = _key_info(score)
        return {
            'time_signature': ts_string,
            'time_signature_idx': TIME_SIGNATURE_IDX[ts_string],
            'key_signature': key_name,
            'key_signature_idx': key_idx,
            'bpm': bpm,
            'tempo_text': tempo_text,
            'num_measures': len(valid_measures),
            'resolution': steps_per_measure,
            'total_length': total_length,
            'num_parts': 2,
            'num_channels': 6 if self.with_velocity else 4,
            'original_measures': original_measures,
            'valid_measures': valid_measures,
            'is_continuation': False,
        }


def save_npz(segments: List[np.ndarray], metadata: Dict[str, Any], output_path: str) -> None:
    payload: Dict[str, Any] = {f'measure_{i}': seg for i, seg in enumerate(segments)}
    payload['metadata'] = metadata
    np.savez_compressed(output_path, **payload)


def _process_one(job) -> Tuple[str, str]:
    xml_path, output_dir, ticks_per_beat, with_velocity, overwrite = job
    out_path = os.path.join(output_dir, Path(xml_path).stem + '.npz')
    if not overwrite and os.path.exists(out_path):
        return xml_path, 'skipped (exists)'
    try:
        segments, metadata = XMLToNPZ(ticks_per_beat, with_velocity).convert(xml_path)
        save_npz(segments, metadata, out_path)
        return xml_path, 'ok'
    except Exception as exc:
        return xml_path, f'failed: {exc}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input', help='a MusicXML file or a directory of them')
    ap.add_argument('--output-dir', required=True, help='where the .npz files go')
    ap.add_argument('--velocity', action='store_true',
                    help='keep per-note velocity (6 channels instead of 4)')
    ap.add_argument('--ticks-per-beat', type=int, default=TICKS_PER_BEAT,
                    help=f'time steps per quarter note (default {TICKS_PER_BEAT}, the paper tau)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        paths = sorted(
            str(p) for p in Path(args.input).rglob('*')
            if p.suffix.lower() in ('.xml', '.mxl', '.musicxml'))
    else:
        paths = [args.input]
    if not paths:
        print(f"no MusicXML files under {args.input}", file=sys.stderr)
        return 1

    jobs = [(p, args.output_dir, args.ticks_per_beat, args.velocity, args.overwrite)
            for p in paths]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_process_one, jobs)
    else:
        results = [_process_one(j) for j in jobs]

    ok = sum(1 for _, status in results if status == 'ok')
    skipped = sum(1 for _, status in results if status.startswith('skipped'))
    failed = [(p, s) for p, s in results if s.startswith('failed')]
    print(f"converted {ok}/{len(paths)}  (skipped {skipped}, failed {len(failed)})")
    for path, status in failed[:20]:
        print(f"  {Path(path).name}: {status[8:]}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
