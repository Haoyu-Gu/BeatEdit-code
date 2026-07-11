#!/usr/bin/env python3
"""Side-by-side demo of the four Beat Encoding schemes.

Encodes a small piece (or your own note spec) under Schemes A-D, prints the
resulting token sequences with statistics, verifies the encode -> decode
round-trip, and optionally writes the decoded piano roll to a MIDI file.

Usage:
    python tools/encoding_demo.py                     # built-in demo piece
    python tools/encoding_demo.py --scheme C          # one scheme only
    python tools/encoding_demo.py --notes notes.json  # your own piece
    python tools/encoding_demo.py --midi-out demo.mid # export decoded MIDI

Note spec format (JSON): a list of beats; each beat is a list of
[pitch_index, pattern] pairs, where pitch_index is 0-87 (A0-C8) and pattern
is the ternary rhythm value 0-80 (paper Appendix A: 1=onset, 2=sustain;
e.g. 53 = quarter note, 80 = continuation of a long note).
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-scheme wiring: tokenizer location, marker token ids, and how
# compress/decompress are parameterized (signatures differ across schemes).
SCHEMES = {
    'A': dict(desc='absolute position, separated tokens',
              markers=dict(track_marker_id=183)),
    'B': dict(desc='relative position, separated tokens',
              markers=dict(empty_marker_id=169, end_marker_id=170)),
    'C': dict(desc='relative position, bundled tokens',
              markers=dict(split_marker_id=7129, empty_marker_id=7128)),
    'D': dict(desc='absolute position, bundled tokens',
              markers=dict(split_marker_id=7129, empty_marker_id=7128)),
}

# Demo piece: C major arpeggio, one chord, and a long note held for two
# beats (the continuation beat encodes as pure-sustain pattern 80).
DEMO_BEATS = [
    [[39, 53]],                        # C4 quarter
    [[43, 53]],                        # E4 quarter
    [[46, 53]],                        # G4 quarter
    [[39, 53], [43, 53], [46, 53]],    # C major chord
    [[51, 53]],                        # C5 quarter, held...
    [[51, 80]],                        # ...sustained through this beat
]


def load_tokenizer(scheme):
    """Load PianoRollTokenizer from src/encoding/scheme_X without packaging."""
    path = os.path.join(ROOT, 'src', 'encoding', f'scheme_{scheme}', 'my_tokenizer.py')
    spec = importlib.util.spec_from_file_location(f'tokenizer_{scheme}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PianoRollTokenizer(patch_h=1, patch_w=4, pattern_num=81, beats_length=88)


def beats_to_pianoroll(beats, tau=4):
    """Convert a note spec into a (2, 88, len(beats)*tau) sustain/onset roll."""
    roll = np.zeros((2, 88, len(beats) * tau), dtype=np.float32)
    for b, notes in enumerate(beats):
        for pitch, pattern in notes:
            if not (0 <= pitch <= 87 and 0 <= pattern <= 80):
                raise ValueError(f'invalid note (pitch={pitch}, pattern={pattern})')
            digits = [(pattern // 27) % 3, (pattern // 9) % 3, (pattern // 3) % 3, pattern % 3]
            for t, d in enumerate(digits):        # 1 = onset, 2 = sustain
                if d:
                    roll[0, pitch, b * tau + t] = 1
                    if d == 1:
                        roll[1, pitch, b * tau + t] = 1
    return roll


def run_scheme(scheme, beats, midi_out=None):
    info = SCHEMES[scheme]
    tok = load_tokenizer(scheme)
    roll = beats_to_pianoroll(beats)

    matrix = tok.image_to_patch_tokens(roll, strict_mode=False)   # (beats, 88) pattern matrix
    compressed = tok.compress_tokens(matrix, **info['markers'])
    restored = tok.decompress_tokens(compressed, **info['markers'])
    roundtrip = np.array_equal(tok.patch_tokens_to_image(restored), roll)

    print(f"\nScheme {scheme} ({info['desc']})")
    print(f"  tokens ({len(compressed)}): {compressed.tolist()}")
    print(f"  tokens/note: {len(compressed) / max(1, sum(len(b) for b in beats)):.2f}   "
          f"round-trip: {'PASS' if roundtrip else 'FAIL'}")

    if midi_out:
        write_midi(tok.patch_tokens_to_image(restored), midi_out)
        print(f"  decoded MIDI written to {midi_out}")
    return roundtrip


def write_midi(roll, path, tempo=120, steps_per_beat=4):
    """Write a (2, 88, T) sustain/onset roll to a single-track MIDI file."""
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    sec = 60.0 / tempo / steps_per_beat
    sustain, onset = roll[0], roll[1]
    for pitch in range(88):
        for start in np.where(onset[pitch] > 0)[0]:
            end = start + 1
            while end < sustain.shape[1] and sustain[pitch, end] > 0 and onset[pitch, end] == 0:
                end += 1
            inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch + 21, start=start * sec, end=end * sec))
    midi.instruments.append(inst)
    midi.write(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scheme', default='all', choices=['A', 'B', 'C', 'D', 'all'])
    ap.add_argument('--notes', help='JSON file with a note spec (see module docstring)')
    ap.add_argument('--midi-out', help='write the decoded piece to this MIDI file')
    args = ap.parse_args()

    beats = DEMO_BEATS
    if args.notes:
        with open(args.notes) as f:
            beats = json.load(f)

    schemes = list(SCHEMES) if args.scheme == 'all' else [args.scheme]
    print(f"Input: {len(beats)} beats, {sum(len(b) for b in beats)} notes")
    ok = all(run_scheme(s, beats, args.midi_out if s == schemes[-1] else None)
             for s in schemes)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
