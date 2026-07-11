"""
Perturbation functions for REMI GECToR training data.

Operates on parsed note structures (bar/position/pitch/velocity/duration).
Perturbations are applied per-note within each bar, analogous to BEAT's per-beat perturbation.

Flow: REMI tokens -> parse -> flat notes -> perturb notes -> reassemble -> new tokens

Key difference from BEAT: insert/delete changes token count (3-4 tokens per note in REMI),
and shared Position tokens complicate insertion/deletion.
"""

import random
from config import (
    MIDI_PITCH_MIN, MIDI_PITCH_MAX,
    VELOCITY_MIN, VELOCITY_MAX,
    DURATION_MIN, DURATION_MAX,
    POSITION_OFFSET, POSITION_MAX, POSITION_MIN,
    PITCH_OFFSET,
    DEFAULT_PERTURB_PROBS, DIFFICULTY_PRESETS,
)
from sequence_parser import (
    parse_sequence, bars_to_flat_notes, flat_notes_to_bars, reassemble_sequence,
)


# ==================== Single-note Perturbations ====================

def perturb_pitch_shift(notes, max_shift=3):
    """
    Randomly shift one note's MIDI pitch by +/-1 to +/-max_shift semitones.

    Args:
        notes: list of (bar_idx, pos_val, midi_pitch, vel_tok, dur_tok)

    Returns: (new_notes, was_changed)
    """
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val, midi_pitch, vel_tok, dur_tok = notes[idx]

    candidates = list(range(-max_shift, 0)) + list(range(1, max_shift + 1))
    shift = random.choice(candidates)
    new_pitch = midi_pitch + shift

    if new_pitch < MIDI_PITCH_MIN or new_pitch > MIDI_PITCH_MAX:
        return notes, False

    # Check for duplicate pitch at same bar+position
    for i, (bi, pv, mp, vt, dt) in enumerate(notes):
        if i != idx and bi == bar_idx and pv == pos_val and mp == new_pitch:
            return notes, False

    new_notes = list(notes)
    new_notes[idx] = (bar_idx, pos_val, new_pitch, vel_tok, dur_tok)
    return new_notes, True


def perturb_rhythm(notes):
    """
    Randomly change one note's duration token.

    Returns: (new_notes, was_changed)
    """
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val, midi_pitch, vel_tok, dur_tok = notes[idx]

    new_dur = dur_tok
    while new_dur == dur_tok:
        new_dur = random.randint(DURATION_MIN, DURATION_MAX)

    new_notes = list(notes)
    new_notes[idx] = (bar_idx, pos_val, midi_pitch, vel_tok, new_dur)
    return new_notes, True


def perturb_delete(notes):
    """
    Randomly delete one note. Won't delete if only 1 note remains in the bar.

    Returns: (new_notes, was_changed)
    """
    if len(notes) <= 1:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx = notes[idx][0]

    # Count notes in same bar
    bar_count = sum(1 for n in notes if n[0] == bar_idx)
    if bar_count <= 1:
        return notes, False

    new_notes = [n for i, n in enumerate(notes) if i != idx]
    return new_notes, True


def perturb_insert(notes):
    """
    Insert a random note at an existing position in a random bar.

    Returns: (new_notes, was_changed)
    """
    if len(notes) == 0:
        return notes, False

    # Pick a random existing note to get bar and position
    ref_idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val = notes[ref_idx][0], notes[ref_idx][1]

    # Find pitches already at this bar+position
    existing_pitches = set(
        mp for bi, pv, mp, vt, dt in notes
        if bi == bar_idx and pv == pos_val
    )

    available = [p for p in range(MIDI_PITCH_MIN, MIDI_PITCH_MAX + 1)
                 if p not in existing_pitches]
    if len(available) == 0:
        return notes, False

    new_pitch = random.choice(available)
    new_vel = random.randint(VELOCITY_MIN, VELOCITY_MAX)
    new_dur = random.randint(DURATION_MIN, DURATION_MAX)

    new_notes = list(notes) + [(bar_idx, pos_val, new_pitch, new_vel, new_dur)]
    return new_notes, True


# ==================== Bar-level Perturbation ====================

def perturb_bar_notes(notes, p_pitch=0.10, p_rhythm=0.05, p_delete=0.03, p_insert=0.02):
    """
    Apply one perturbation to a bar's notes (mutually exclusive, sampled by probability).

    Args:
        notes: list of (bar_idx, pos_val, midi_pitch, vel_tok, dur_tok) for ONE bar

    Returns: (perturbed_notes, was_changed)
    """
    r = random.random()
    cumulative = 0.0

    cumulative += p_pitch
    if r < cumulative:
        return perturb_pitch_shift(notes)

    cumulative += p_rhythm
    if r < cumulative:
        return perturb_rhythm(notes)

    cumulative += p_delete
    if r < cumulative:
        return perturb_delete(notes)

    cumulative += p_insert
    if r < cumulative:
        return perturb_insert(notes)

    return notes, False


# ==================== Sequence-level Perturbation ====================

def perturb_sequence(token_sequence, p_pitch=0.10, p_rhythm=0.05,
                     p_delete=0.03, p_insert=0.02):
    """
    Perturb a complete REMI token sequence by independently perturbing each bar.

    Args:
        token_sequence: original clean REMI token sequence (list of ints)

    Returns:
        (source_tokens, target_tokens) where:
        - source = perturbed sequence (model input)
        - target = original sequence (correct answer)
    """
    parsed = parse_sequence(token_sequence)
    all_notes = bars_to_flat_notes(parsed)
    num_bars = len(parsed['bars'])

    if num_bars == 0 or len(all_notes) == 0:
        return list(token_sequence), list(token_sequence)

    # Group notes by bar
    bar_notes = {}
    for note in all_notes:
        bar_idx = note[0]
        if bar_idx not in bar_notes:
            bar_notes[bar_idx] = []
        bar_notes[bar_idx].append(note)

    # Perturb each bar independently
    perturbed_notes = []
    for bar_idx in range(num_bars):
        notes = bar_notes.get(bar_idx, [])
        if len(notes) == 0:
            continue
        new_notes, _ = perturb_bar_notes(
            notes,
            p_pitch=p_pitch,
            p_rhythm=p_rhythm,
            p_delete=p_delete,
            p_insert=p_insert,
        )
        perturbed_notes.extend(new_notes)

    # Reassemble
    new_bars = flat_notes_to_bars(perturbed_notes, num_bars)
    source_tokens = reassemble_sequence(parsed, new_bars)
    target_tokens = list(token_sequence)

    return source_tokens, target_tokens


def perturb_sequence_per_position(token_sequence, p_pitch=0.10, p_rhythm=0.05,
                                   p_delete=0.03, p_insert=0.02):
    """
    Perturb at Position level (matching BEAT's per-beat perturbation).

    Each Position within each bar gets an independent perturbation chance,
    analogous to BEAT encoding where each beat is independently perturbed.
    This produces ~4x more perturbations than bar-level perturbation since
    a typical bar has ~4 positions.

    Use this for fair comparison with BEAT GECToR results.
    """
    parsed = parse_sequence(token_sequence)
    all_notes = bars_to_flat_notes(parsed)
    num_bars = len(parsed['bars'])

    if num_bars == 0 or len(all_notes) == 0:
        return list(token_sequence), list(token_sequence)

    # Group notes by (bar_idx, position_value)
    pos_notes = {}
    for note in all_notes:
        key = (note[0], note[1])  # (bar_idx, position_value)
        if key not in pos_notes:
            pos_notes[key] = []
        pos_notes[key].append(note)

    # Perturb each position independently (like BEAT perturbs each beat)
    perturbed_notes = []
    for key in sorted(pos_notes.keys()):
        notes = pos_notes[key]
        new_notes, _ = perturb_bar_notes(
            notes,
            p_pitch=p_pitch,
            p_rhythm=p_rhythm,
            p_delete=p_delete,
            p_insert=p_insert,
        )
        perturbed_notes.extend(new_notes)

    # Reassemble
    new_bars = flat_notes_to_bars(perturbed_notes, num_bars)
    source_tokens = reassemble_sequence(parsed, new_bars)
    target_tokens = list(token_sequence)

    return source_tokens, target_tokens


def perturb_sequence_with_difficulty(token_sequence, difficulty='medium'):
    """Perturb using preset difficulty level."""
    probs = DIFFICULTY_PRESETS[difficulty]
    return perturb_sequence(
        token_sequence,
        p_pitch=probs['p_pitch'],
        p_rhythm=probs['p_rhythm'],
        p_delete=probs['p_delete'],
        p_insert=probs['p_insert'],
    )
