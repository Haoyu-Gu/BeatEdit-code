"""
Perturbation functions for CPWord GECToR training data.

Operates on parsed note structures (bar/position/pitch/velocity/duration).
Perturbations are applied per-position within each bar, matching BEAT's per-beat perturbation.

Flow: compound tokens -> parse -> flat notes -> perturb notes -> reassemble -> new tokens

Key advantage over REMI: each note is 1 compound token, so insert/delete is simpler.
"""

import random
from config import (
    MIDI_PITCH_MIN, MIDI_PITCH_MAX,
    VEL_OFFSET, VEL_MAX_ID,
    DUR_OFFSET, DUR_MAX_ID,
    DEFAULT_PERTURB_PROBS,
    BOS_TOKEN, EOS_TOKEN,
    is_special_token,
)
from sequence_parser import (
    parse_sequence, bars_to_flat_notes, flat_notes_to_bars, reassemble_sequence,
)


# ==================== Single-note Perturbations ====================

def perturb_pitch_shift(notes, max_shift=3):
    """Randomly shift one note's MIDI pitch by +/-1 to +/-max_shift semitones."""
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val, midi_pitch, vel_id, dur_id = notes[idx]

    candidates = list(range(-max_shift, 0)) + list(range(1, max_shift + 1))
    shift = random.choice(candidates)
    new_pitch = midi_pitch + shift

    if new_pitch < MIDI_PITCH_MIN or new_pitch > MIDI_PITCH_MAX:
        return notes, False

    # Check for duplicate pitch at same bar+position
    for i, (bi, pv, mp, vi, di) in enumerate(notes):
        if i != idx and bi == bar_idx and pv == pos_val and mp == new_pitch:
            return notes, False

    new_notes = list(notes)
    new_notes[idx] = (bar_idx, pos_val, new_pitch, vel_id, dur_id)
    return new_notes, True


def perturb_rhythm(notes):
    """Randomly change one note's duration sub-token."""
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val, midi_pitch, vel_id, dur_id = notes[idx]

    new_dur = dur_id
    while new_dur == dur_id:
        new_dur = random.randint(DUR_OFFSET, DUR_MAX_ID)

    new_notes = list(notes)
    new_notes[idx] = (bar_idx, pos_val, midi_pitch, vel_id, new_dur)
    return new_notes, True


def perturb_delete(notes):
    """Randomly delete one note. Won't delete if only 1 note in the bar."""
    if len(notes) <= 1:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    bar_idx = notes[idx][0]

    bar_count = sum(1 for n in notes if n[0] == bar_idx)
    if bar_count <= 1:
        return notes, False

    new_notes = [n for i, n in enumerate(notes) if i != idx]
    return new_notes, True


def perturb_insert(notes):
    """Insert a random note at an existing position in a random bar."""
    if len(notes) == 0:
        return notes, False

    ref_idx = random.randint(0, len(notes) - 1)
    bar_idx, pos_val = notes[ref_idx][0], notes[ref_idx][1]

    existing_pitches = set(
        mp for bi, pv, mp, vi, di in notes
        if bi == bar_idx and pv == pos_val
    )

    available = [p for p in range(MIDI_PITCH_MIN, MIDI_PITCH_MAX + 1)
                 if p not in existing_pitches]
    if len(available) == 0:
        return notes, False

    new_pitch = random.choice(available)
    new_vel = random.randint(VEL_OFFSET, VEL_MAX_ID)
    new_dur = random.randint(DUR_OFFSET, DUR_MAX_ID)

    new_notes = list(notes) + [(bar_idx, pos_val, new_pitch, new_vel, new_dur)]
    return new_notes, True


# ==================== Position-level Perturbation ====================

def perturb_position_notes(notes, p_pitch=0.10, p_rhythm=0.05, p_delete=0.03, p_insert=0.02):
    """Apply one perturbation to a position's notes (sampled by probability)."""
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
    Perturb a CPWord compound token sequence at Position level.

    Each Position within each bar gets an independent perturbation chance,
    matching BEAT encoding's per-beat perturbation.

    Args:
        token_sequence: original clean compound token sequence (list of [5])

    Returns:
        (source_tokens, target_tokens) where:
        - source = perturbed sequence (model input)
        - target = original sequence (correct answer)
    """
    parsed = parse_sequence(token_sequence)
    all_notes = bars_to_flat_notes(parsed)
    num_bars = len(parsed['bars'])

    if num_bars == 0 or len(all_notes) == 0:
        return [list(t) for t in token_sequence], [list(t) for t in token_sequence]

    # Group notes by (bar_idx, position_value)
    pos_notes = {}
    for note in all_notes:
        key = (note[0], note[1])
        if key not in pos_notes:
            pos_notes[key] = []
        pos_notes[key].append(note)

    # Perturb each position independently
    perturbed_notes = []
    for key in sorted(pos_notes.keys()):
        notes = pos_notes[key]
        new_notes, _ = perturb_position_notes(
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
    target_tokens = [list(t) for t in token_sequence]

    return source_tokens, target_tokens
