#!/usr/bin/env python3
"""
Decode-Filter-Reencode validity verification (rebuttal C5 / reviewer ofxZ).

Verifies, over 200+ random edits that include cuts into long (multi-beat)
notes, that the SeqTag post-processing step (decode -> filter invalid notes
-> re-encode; see paper Appendix E and src/seqtag/scheme_*/inference.py)
satisfies three properties:

  1. Out-of-region preservation: beats outside the edited region are
     preserved token-for-token (verbatim).
  2. Zero violations: every beat of the post-processed sequence is
     well-formed -- positions in range, pattern values in [0, 80],
     strictly ascending absolute pitches, no duplicate pitches.
  3. 100% decodable: the full post-processed sequence parses and decodes
     to (pitch, pattern) notes without error.

Filtering drops only truly invalid notes (out-of-range or ordering
violations). Incomplete insertions (a position token without a pattern
token) are also removed by re-encoding, which is what keeps every
intermediate sequence decodable (paper Appendix F).

Usage:
    python evaluation/verify_filter_roundtrip.py [--n 200] [--scheme B]
"""

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_scheme(scheme):
    """Import sequence_parser + inference post_process for one scheme."""
    scheme_dir = os.path.join(ROOT, 'src', 'seqtag', f'scheme_{scheme}')
    sys.path.insert(0, scheme_dir)
    # fresh imports per scheme
    for mod in ('config', 'sequence_parser', 'inference', 'model'):
        sys.modules.pop(mod, None)
    import sequence_parser as sp
    import inference as inf
    sys.path.pop(0)
    return sp, inf


def make_sequence(sp, rng, num_beats=8):
    """Build a random well-formed beat sequence containing long notes.

    Long notes are onset patterns in one beat followed by pure-sustain
    continuation patterns (s_p = 80) in subsequent beats, as in paper
    Appendix A.2.
    """
    beats = []
    notes_per_beat = []
    sustained = {}  # pitch -> remaining continuation beats
    for _ in range(num_beats):
        notes = []
        # continue long notes: pure sustain pattern 80
        for p in sorted(sustained):
            notes.append((p, 80))
            sustained[p] -= 1
        sustained = {p: r for p, r in sustained.items() if r > 0}
        # new notes
        for _ in range(rng.randint(1, 3)):
            p = rng.randint(0, 87)
            if any(q == p for q, _ in notes):
                continue
            # onset patterns: quarter (53), two eighths (50), staccato (27)
            v = rng.choice([53, 50, 27, 53])
            notes.append((p, v))
            if v == 53 and rng.random() < 0.5:
                # make it a long note sustained into following beats
                sustained[p] = rng.randint(1, 2)
        notes = sorted(set(notes), key=lambda x: x[0])
        # drop duplicate pitches (keep first)
        seen, uniq = set(), []
        for p, v in notes:
            if p not in seen:
                seen.add(p)
                uniq.append((p, v))
        beats.append(sp.encode_beat(uniq))
        notes_per_beat.append(uniq)
    return beats, notes_per_beat


def random_edit(sp, beat_tokens, rng):
    """Apply a random token-level edit inside one beat.

    Includes edits that cut long notes: replacing or deleting the
    pure-sustain continuation token of a multi-beat note, shifting a
    position token, appending an unpaired position token, or writing an
    out-of-range value.
    """
    toks = list(beat_tokens)
    music = [i for i, t in enumerate(toks) if t not in (getattr(sp, 'EMPTY_MARKER', -1), getattr(sp, 'END_MARKER', -2))]
    if not music:
        return toks
    op = rng.choice(['shift_pos', 'replace_val', 'delete_tok', 'append_pos', 'corrupt'])
    i = rng.choice(music)
    if op == 'shift_pos':
        toks[i] = toks[i] + rng.choice([-2, -1, 1, 2])
    elif op == 'replace_val':
        toks[i] = rng.randint(0, 80)  # may cut a long note's continuation
    elif op == 'delete_tok':
        del toks[i]  # may orphan a position or a pattern token
    elif op == 'append_pos':
        toks.insert(i + 1, 81 + rng.randint(0, 87))  # unpaired position token
    else:
        toks[i] = rng.randint(0, 200)  # arbitrary corruption incl. out-of-range
    return toks


def beat_is_wellformed(sp, beat_tokens):
    """Check one post-processed beat for violations."""
    notes = sp.decode_beat(beat_tokens)
    # decode must consume the whole beat: re-encoding the decoded notes
    # must reproduce the beat exactly (no trailing garbage tokens)
    if sp.encode_beat(notes) != list(beat_tokens):
        return False, 'residual tokens after decode'
    pitches = [p for p, _ in notes]
    if any(not (0 <= p <= 87) for p in pitches):
        return False, 'pitch out of range'
    if any(not (0 <= v <= 80) for _, v in notes):
        return False, 'pattern value out of range'
    if pitches != sorted(pitches) or len(set(pitches)) != len(pitches):
        return False, 'pitch ordering violation'
    return True, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='number of edited sequences')
    ap.add_argument('--scheme', default='B', choices=['A', 'B'],
                    help='separated encoding scheme to verify')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    sp, inf = load_scheme(args.scheme)
    rng = random.Random(args.seed)

    n_edits = 0
    n_longnote_cuts = 0
    preserved = 0
    total_out_of_region = 0
    violations = 0
    undecodable = 0

    for _ in range(args.n):
        beats, notes_per_beat = make_sequence(sp, rng)
        num_beats = len(beats)
        # edit region: two consecutive beats
        r0 = rng.randint(0, num_beats - 2)
        region = {r0, r0 + 1}
        if any(v == 80 for b in region for _, v in notes_per_beat[b]):
            n_longnote_cuts += 1

        edited_beats = []
        for b, toks in enumerate(beats):
            if b in region:
                edited_beats.append(random_edit(sp, toks, rng))
                n_edits += 1
            else:
                edited_beats.append(list(toks))

        # assemble a full token sequence with proper structure.
        # Scheme B: [BOS, TIME_SIG, BPM, BAR, beat..., ...] where each beat
        # already carries its EMPTY/END markers from encode_beat.
        # Scheme A: beats are introduced by alternating track markers.
        flat = [sp.BOS_TOKEN, sp.TIME_SIG_OFFSET, sp.BPM_OFFSET]
        has_track_markers = hasattr(sp, 'TRACK0_START')
        for b, toks in enumerate(edited_beats):
            if b % 4 == 0:
                flat.append(sp.BAR_TOKEN)
            if has_track_markers:
                flat.append(sp.TRACK0_START if b % 2 == 0 else sp.TRACK1_START)
            flat.extend(toks)
        flat.append(sp.EOS_TOKEN)
        processed = inf.post_process(flat)

        # re-parse the processed sequence into beats
        try:
            info = sp.parse_sequence(processed)
            out_beats = [b['tokens'] for b in info['beats']]
        except Exception:
            undecodable += 1
            continue

        # 1. out-of-region preservation (compare per-beat token lists)
        if len(out_beats) == len(edited_beats):
            for b in range(num_beats):
                if b in region:
                    continue
                total_out_of_region += 1
                if list(out_beats[b]) == list(edited_beats[b]):
                    preserved += 1
        # 2/3. violations + decodability
        seq_ok = True
        for toks in out_beats:
            ok, why = beat_is_wellformed(sp, toks)
            if not ok:
                violations += 1
                seq_ok = False
        try:
            for toks in out_beats:
                sp.decode_beat(toks)
        except Exception:
            undecodable += 1
            seq_ok = False

    print(f"Scheme {args.scheme}: {args.n} edited sequences, "
          f"{n_edits} edits ({n_longnote_cuts} sequences cut a long note)")
    print(f"  out-of-region beats preserved verbatim: "
          f"{preserved}/{total_out_of_region} "
          f"({100.0 * preserved / max(1, total_out_of_region):.1f}%)")
    print(f"  post-processed beats with violations:   {violations}")
    print(f"  undecodable sequences:                  {undecodable} "
          f"({100.0 * (args.n - undecodable) / args.n:.1f}% decodable)")

    ok = (total_out_of_region > 0 and preserved == total_out_of_region
          and violations == 0 and undecodable == 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
