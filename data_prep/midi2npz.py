#!/usr/bin/env python3
"""MIDI -> piano NPZ converter (same output layout as ``xml2npz.py``).

Use this for MIDI sources (e.g. the Lakh MIDI Dataset used in the multi-track
experiments); use ``xml2npz.py`` for the MuseScore MusicXML corpus, which is
what the main piano experiments are trained on.

MIDI has no notion of a treble/bass staff, so the two voices are recovered
heuristically:

  * ``--split tracks`` (default when the file has >= 2 non-empty instrument
    tracks): the first two non-drum instruments become treble and bass, ordered
    by mean pitch (higher = treble).
  * ``--split pitch``: a single flattened track is split at a pitch threshold
    (default MIDI 60, middle C).

Everything else -- the tau = 4 sixteenth-note grid, the 88-key ascending pitch
axis, the 4- or 6-channel layout and the metadata schema -- matches
``xml2npz.py``; see that module's docstring for the full format.

Usage
-----
    python data_prep/midi2npz.py data/lakh/ --output-dir data/npz --workers 8
    python data_prep/midi2npz.py song.mid --output-dir data/npz --velocity
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    sys.exit("pretty_midi is required: pip install pretty_midi")

from xml2npz import (  # noqa: E402  (same directory)
    DEFAULT_VELOCITY, MAX_MEASURES, MAX_PITCH, MIN_MEASURES, MIN_PITCH,
    PITCH_RANGE, TICKS_PER_BEAT, TIME_SIGNATURE_IDX, save_npz,
)

DEFAULT_SPLIT_PITCH = 60  # middle C


def _time_signature(midi: 'pretty_midi.PrettyMIDI') -> str:
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        return f"{ts.numerator}/{ts.denominator}"
    return '4/4'


def _tempo(midi: 'pretty_midi.PrettyMIDI') -> float:
    try:
        _, tempi = midi.get_tempo_changes()
        if len(tempi):
            return float(tempi[0])
    except Exception:
        pass
    return 120.0


def _split_voices(midi, mode: str, split_pitch: int) -> Tuple[List, List]:
    """Return (treble_notes, bass_notes) as lists of pretty_midi Note objects."""
    tracks = [inst for inst in midi.instruments if inst.notes and not inst.is_drum]
    if not tracks:
        raise ValueError("no pitched notes")

    if mode == 'tracks' and len(tracks) >= 2:
        tracks.sort(key=lambda t: np.mean([n.pitch for n in t.notes]), reverse=True)
        return tracks[0].notes, tracks[1].notes

    notes = [n for t in tracks for n in t.notes]
    treble = [n for n in notes if n.pitch >= split_pitch]
    bass = [n for n in notes if n.pitch < split_pitch]
    if not treble or not bass:
        raise ValueError(f"pitch split at {split_pitch} left one voice empty")
    return treble, bass


def _notes_to_rolls(notes, total_steps: int, sec_per_step: float):
    sustain = np.zeros((PITCH_RANGE, total_steps), dtype=np.uint8)
    onset = np.zeros((PITCH_RANGE, total_steps), dtype=np.uint8)
    velocity = np.zeros((PITCH_RANGE, total_steps), dtype=np.uint8)
    for note in notes:
        if not (MIN_PITCH <= note.pitch <= MAX_PITCH):
            continue
        start = int(round(note.start / sec_per_step))
        end = max(start + 1, int(round(note.end / sec_per_step)))
        if start >= total_steps:
            continue
        end = min(end, total_steps)
        p = note.pitch - MIN_PITCH
        sustain[p, start:end] = 1
        onset[p, start] = 1
        velocity[p, start:end] = note.velocity or DEFAULT_VELOCITY
    return sustain, onset, velocity


def convert(midi_path: str, with_velocity: bool = False,
            split: str = 'tracks', split_pitch: int = DEFAULT_SPLIT_PITCH,
            ticks_per_beat: int = TICKS_PER_BEAT) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    midi = pretty_midi.PrettyMIDI(midi_path)

    ts_string = _time_signature(midi)
    if ts_string not in TIME_SIGNATURE_IDX:
        raise ValueError(f"time signature {ts_string} not representable")

    bpm = _tempo(midi)
    beats_per_measure = int(ts_string.split('/')[0])
    if ts_string == '6/8':
        beats_per_measure = 3          # 6/8 is felt in 3 quarter-note beats
    elif ts_string == '2/2':
        beats_per_measure = 4
    steps_per_measure = beats_per_measure * ticks_per_beat

    sec_per_beat = 60.0 / bpm
    sec_per_step = sec_per_beat / ticks_per_beat
    total_steps = int(np.ceil(midi.get_end_time() / sec_per_step))
    num_measures = total_steps // steps_per_measure
    if not (MIN_MEASURES <= num_measures <= MAX_MEASURES):
        raise ValueError(
            f"measure count {num_measures} outside [{MIN_MEASURES}, {MAX_MEASURES}]")
    total_steps = num_measures * steps_per_measure

    treble_notes, bass_notes = _split_voices(midi, split, split_pitch)
    treble = _notes_to_rolls(treble_notes, total_steps, sec_per_step)
    bass = _notes_to_rolls(bass_notes, total_steps, sec_per_step)

    segments = []
    for m in range(num_measures):
        s, e = m * steps_per_measure, (m + 1) * steps_per_measure
        if with_velocity:
            seg = np.stack([treble[0][:, s:e], treble[1][:, s:e], treble[2][:, s:e],
                            bass[0][:, s:e], bass[1][:, s:e], bass[2][:, s:e]], axis=0)
        else:
            seg = np.stack([treble[0][:, s:e], treble[1][:, s:e],
                            bass[0][:, s:e], bass[1][:, s:e]], axis=0)
        segments.append(seg.astype(np.uint8))

    metadata = {
        'time_signature': ts_string,
        'time_signature_idx': TIME_SIGNATURE_IDX[ts_string],
        'key_signature': None,
        'key_signature_idx': 0,
        'bpm': float(bpm),
        'tempo_text': None,
        'num_measures': num_measures,
        'resolution': steps_per_measure,
        'total_length': total_steps,
        'num_parts': 2,
        'num_channels': 6 if with_velocity else 4,
        'original_measures': num_measures,
        'valid_measures': list(range(num_measures)),
        'is_continuation': False,
    }
    return segments, metadata


def _process_one(job) -> Tuple[str, str]:
    path, output_dir, with_velocity, split, split_pitch, overwrite = job
    out = os.path.join(output_dir, Path(path).stem + '.npz')
    if not overwrite and os.path.exists(out):
        return path, 'skipped (exists)'
    try:
        segments, metadata = convert(path, with_velocity, split, split_pitch)
        save_npz(segments, metadata, out)
        return path, 'ok'
    except Exception as exc:
        return path, f'failed: {exc}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input', help='a MIDI file or a directory of them')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--velocity', action='store_true',
                    help='keep per-note velocity (6 channels instead of 4)')
    ap.add_argument('--split', choices=['tracks', 'pitch'], default='tracks',
                    help='how to recover the two voices (default: tracks)')
    ap.add_argument('--split-pitch', type=int, default=DEFAULT_SPLIT_PITCH,
                    help='pitch threshold for --split pitch (default 60 = middle C)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isdir(args.input):
        paths = sorted(str(p) for p in Path(args.input).rglob('*')
                       if p.suffix.lower() in ('.mid', '.midi'))
    else:
        paths = [args.input]
    if not paths:
        print(f"no MIDI files under {args.input}", file=sys.stderr)
        return 1

    jobs = [(p, args.output_dir, args.velocity, args.split, args.split_pitch, args.overwrite)
            for p in paths]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_process_one, jobs)
    else:
        results = [_process_one(j) for j in jobs]

    ok = sum(1 for _, s in results if s == 'ok')
    failed = [(p, s) for p, s in results if s.startswith('failed')]
    print(f"converted {ok}/{len(paths)}  (failed {len(failed)})")
    for path, status in failed[:20]:
        print(f"  {Path(path).name}: {status[8:]}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
