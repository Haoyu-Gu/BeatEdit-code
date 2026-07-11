"""
Perturbation functions for GECToR training data (no_pair encoding).

Core principle: perturb in absolute pitch space, then re-encode to absolute positions.
Since Scheme A uses absolute positions, re-encoding is simpler than relative positions.

Flow: original tokens -> parse -> decode beats to notes -> perturb notes
      -> re-encode beats -> reassemble sequence
"""

import random
from config import (
    MAX_PITCH, PATCH_VALUE_MIN, PATCH_VALUE_MAX,
    DEFAULT_PERTURB_PROBS, DIFFICULTY_PRESETS,
)
from sequence_parser import decode_beat, encode_beat, parse_sequence, reassemble_sequence


# ==================== Single-note Perturbations ====================

def perturb_pitch_shift(notes, max_shift=3):
    """
    Randomly shift one note's pitch by +/-1 to +/-max_shift semitones.

    Constraints:
    - New pitch must be in [0, 87]
    - New pitch must not duplicate another note in the beat

    Returns: (new_notes, was_changed)
    """
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    pitch, val = notes[idx]

    candidates = list(range(-max_shift, 0)) + list(range(1, max_shift + 1))
    shift = random.choice(candidates)
    new_pitch = pitch + shift

    if new_pitch < 0 or new_pitch > MAX_PITCH:
        return notes, False
    if any(p == new_pitch for i, (p, v) in enumerate(notes) if i != idx):
        return notes, False

    new_notes = list(notes)
    new_notes[idx] = (new_pitch, val)
    return new_notes, True


def perturb_rhythm(notes):
    """
    Randomly change one note's patch value (rhythm pattern).

    Returns: (new_notes, was_changed)
    """
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    pitch, val = notes[idx]

    new_val = val
    while new_val == val:
        new_val = random.randint(PATCH_VALUE_MIN, PATCH_VALUE_MAX)

    new_notes = list(notes)
    new_notes[idx] = (pitch, new_val)
    return new_notes, True


def perturb_delete(notes):
    """
    Randomly delete one note from the beat.
    Won't delete if the beat has only 1 note (to avoid creating empty beats).

    Returns: (new_notes, was_changed)
    """
    if len(notes) <= 1:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    new_notes = [n for i, n in enumerate(notes) if i != idx]
    return new_notes, True


def perturb_insert(notes):
    """
    Insert a random note at an unused pitch position.

    Returns: (new_notes, was_changed)
    """
    existing_pitches = set(p for p, v in notes)
    available = [p for p in range(MAX_PITCH + 1) if p not in existing_pitches]

    if len(available) == 0:
        return notes, False

    new_pitch = random.choice(available)
    new_val = random.randint(PATCH_VALUE_MIN, PATCH_VALUE_MAX)

    new_notes = list(notes) + [(new_pitch, new_val)]
    return new_notes, True


# ==================== Beat-level Perturbation ====================

def perturb_beat(notes, p_pitch=0.10, p_rhythm=0.05, p_delete=0.03, p_insert=0.02):
    """
    Apply one perturbation to a beat (mutually exclusive, sampled by probability).

    Args:
        notes: list of (abs_pitch, patch_value) tuples
        p_pitch: probability of pitch shift
        p_rhythm: probability of rhythm change
        p_delete: probability of note deletion
        p_insert: probability of note insertion

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

    # No perturbation
    return notes, False


# ==================== Sequence-level Perturbation ====================

def perturb_sequence(token_sequence, p_pitch=0.10, p_rhythm=0.05,
                     p_delete=0.03, p_insert=0.02):
    """
    Perturb a complete token sequence by independently perturbing each beat.

    Args:
        token_sequence: original clean token sequence (list of ints)
        p_*: perturbation probabilities per beat

    Returns:
        (source_tokens, target_tokens) where:
        - source = perturbed sequence (with errors, model input)
        - target = original sequence (correct answer)
    """
    beats_info = parse_sequence(token_sequence)

    new_beats = []
    num_perturbed = 0

    for beat in beats_info['beats']:
        notes = decode_beat(beat['tokens'])
        # Skip empty beats: they have no note info,
        # inserting notes would produce malformed sequences
        if len(notes) == 0:
            new_beats.append(beat['tokens'])
            continue
        perturbed_notes, changed = perturb_beat(
            notes,
            p_pitch=p_pitch,
            p_rhythm=p_rhythm,
            p_delete=p_delete,
            p_insert=p_insert,
        )
        new_beats.append(encode_beat(perturbed_notes))
        if changed:
            num_perturbed += 1

    source_tokens = reassemble_sequence(beats_info, new_beats)
    target_tokens = list(token_sequence)

    return source_tokens, target_tokens


def perturb_sequence_with_difficulty(token_sequence, difficulty='medium'):
    """
    Perturb using preset difficulty level.

    Args:
        token_sequence: original clean token sequence
        difficulty: 'easy', 'medium', or 'hard'

    Returns:
        (source_tokens, target_tokens)
    """
    probs = DIFFICULTY_PRESETS[difficulty]
    return perturb_sequence(
        token_sequence,
        p_pitch=probs['p_pitch'],
        p_rhythm=probs['p_rhythm'],
        p_delete=probs['p_delete'],
        p_insert=probs['p_insert'],
    )
