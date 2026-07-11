"""
FELIX-Music Perturbation System (Scheme A: no_pair).

4-level perturbation that operates ONLY on accompaniment (Track 1) beats.
Melody (Track 0) is never modified.

Key differences from Scheme C (with_pair):
- No SPLIT markers; beats use TRACK markers
- Empty beats: tokens=[0], not tokens=[]
- encode_beat returns [0] for empty, [pos,val,...] for non-empty (absolute positions)

Level 1 (5-15%): light note-level edits (pitch shift, rhythm change, note add/delete)
Level 2 (15-40%): Level 1 + passage transposition + note reduction
Level 3 (40-70%): Level 2 + whole-beat deletion + simplification
Level 4 (100%): complete accompaniment wipe (all empty beats)
"""

import random
import copy

from data.sequence_parser import decode_beat, encode_beat
from configs.config import (
    MAX_PITCH, PATTERN_NUM,
)


def _is_empty_beat(beat):
    """Check if a beat is empty."""
    tokens = beat['tokens']
    return len(tokens) == 0 or (len(tokens) == 1 and tokens[0] == 0)


def _make_empty_beat(beat):
    """Create an empty beat dict."""
    return {
        'tokens': [0],
        'track_id': beat.get('track_id'),
        'start_idx': beat.get('start_idx', 0),
        'end_idx': beat.get('end_idx', 0),
    }


# ==================== Single-note Perturbation Primitives ====================

def perturb_pitch_shift(notes, max_shift=3):
    """Randomly shift one note's pitch by +/-1 to +/-max_shift semitones."""
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
    """Randomly change one note's patch value (rhythm pattern)."""
    if len(notes) == 0:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    pitch, val = notes[idx]

    new_val = val
    while new_val == val:
        new_val = random.randint(0, PATTERN_NUM - 1)

    new_notes = list(notes)
    new_notes[idx] = (pitch, new_val)
    return new_notes, True


def perturb_delete_note(notes):
    """Randomly delete one note from the beat."""
    if len(notes) <= 1:
        return notes, False

    idx = random.randint(0, len(notes) - 1)
    new_notes = [n for i, n in enumerate(notes) if i != idx]
    return new_notes, True


def perturb_insert_note(notes):
    """Insert a random note at an unused pitch position."""
    existing_pitches = set(p for p, v in notes)
    available = [p for p in range(MAX_PITCH + 1) if p not in existing_pitches]

    if len(available) == 0:
        return notes, False

    new_pitch = random.choice(available)
    new_val = random.randint(0, PATTERN_NUM - 1)

    new_notes = list(notes) + [(new_pitch, new_val)]
    return new_notes, True


# ==================== Beat-level Perturbation Helpers ====================

def _apply_note_level_perturbation(notes):
    """Apply one random note-level perturbation to a beat's notes."""
    r = random.random()
    if r < 0.35:
        return perturb_pitch_shift(notes)
    elif r < 0.60:
        return perturb_rhythm(notes)
    elif r < 0.80:
        return perturb_delete_note(notes)
    else:
        return perturb_insert_note(notes)


def _transpose_notes(notes, semitones):
    """Transpose all notes in a beat by given semitones."""
    new_notes = []
    for pitch, val in notes:
        new_pitch = pitch + semitones
        if 0 <= new_pitch <= MAX_PITCH:
            new_notes.append((new_pitch, val))
    return new_notes


def _simplify_beat(notes):
    """Simplify a beat to just root note with a simple pattern (val=54)."""
    if len(notes) == 0:
        return []
    # Keep lowest pitch as root, set pattern to 54 (a common simple pattern)
    root_pitch = min(p for p, v in notes)
    return [(root_pitch, 54)]


def _generate_random_beat():
    """Generate a simple random beat (1-3 notes) to fill empty beats."""
    n_notes = random.randint(1, 3)
    pitches = random.sample(range(20, 70), n_notes)  # mid-range pitches
    return [(p, random.choice([54, 27, 36, 0])) for p in sorted(pitches)]


# ==================== 4-Level Perturbation System ====================

def _perturb_level_1(accomp_beats):
    """
    Level 1: Light perturbation (5-15% of beats).
    Operations: pitch shift, rhythm change, note add/delete on selected beats.
    """
    n = len(accomp_beats)
    if n == 0:
        return accomp_beats, []

    edit_ratio = random.uniform(0.08, 0.20)
    num_edits = max(1, int(n * edit_ratio))
    edit_indices = set(random.sample(range(n), min(num_edits, n)))

    result = []
    changed_mask = [False] * n

    for i, beat in enumerate(accomp_beats):
        if i in edit_indices:
            notes = decode_beat(beat['tokens'])
            if len(notes) == 0:
                result.append(copy.deepcopy(beat))
                continue
            new_notes, changed = _apply_note_level_perturbation(notes)
            if changed:
                new_tokens = encode_beat(new_notes)
                new_beat = copy.deepcopy(beat)
                new_beat['tokens'] = new_tokens
                result.append(new_beat)
                changed_mask[i] = True
            else:
                result.append(copy.deepcopy(beat))
        else:
            result.append(copy.deepcopy(beat))

    return result, changed_mask


def _perturb_level_2(accomp_beats):
    """
    Level 2: Moderate perturbation (15-40% of beats).
    Level 1 operations + passage transposition + note reduction.
    """
    n = len(accomp_beats)
    if n == 0:
        return accomp_beats, []

    # Start with level 1
    result, changed_mask = _perturb_level_1(accomp_beats)

    # Passage transposition: select a contiguous passage (3-8 beats) and transpose
    if n >= 3:
        passage_len = random.randint(3, min(8, n))
        start = random.randint(0, n - passage_len)
        semitones = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])

        for i in range(start, start + passage_len):
            notes = decode_beat(result[i]['tokens'])
            if len(notes) == 0:
                continue
            new_notes = _transpose_notes(notes, semitones)
            if len(new_notes) > 0:
                new_tokens = encode_beat(new_notes)
                result[i]['tokens'] = new_tokens
                changed_mask[i] = True

    # Fill some empty beats with random content (enables DELETE labels)
    fill_ratio = random.uniform(0.03, 0.08)
    num_fills = max(1, int(n * fill_ratio))
    empty_indices = [i for i in range(n) if _is_empty_beat(result[i])]
    if empty_indices:
        fill_count = min(num_fills, len(empty_indices))
        fill_indices = random.sample(empty_indices, fill_count)
        for i in fill_indices:
            new_notes = _generate_random_beat()
            new_tokens = encode_beat(new_notes)
            result[i]['tokens'] = new_tokens
            changed_mask[i] = True

    # Additional note reduction on some beats
    extra_ratio = random.uniform(0.08, 0.20)
    extra_edits = max(1, int(n * extra_ratio))
    extra_indices = random.sample(range(n), min(extra_edits, n))

    for i in extra_indices:
        if changed_mask[i]:
            continue
        notes = decode_beat(result[i]['tokens'])
        if len(notes) > 1:
            # Remove a random subset of notes
            keep_count = max(1, len(notes) // 2)
            kept = random.sample(notes, keep_count)
            new_tokens = encode_beat(kept)
            result[i]['tokens'] = new_tokens
            changed_mask[i] = True

    return result, changed_mask


def _perturb_level_3(accomp_beats):
    """
    Level 3: Heavy perturbation (40-70% of beats).
    Level 2 operations + whole-beat deletion + simplification to root note.
    """
    n = len(accomp_beats)
    if n == 0:
        return accomp_beats, []

    # Start with level 2
    result, changed_mask = _perturb_level_2(accomp_beats)

    # Fill some empty beats with random content (enables DELETE labels)
    fill_ratio = random.uniform(0.05, 0.12)
    num_fills = max(1, int(n * fill_ratio))
    empty_indices = [i for i in range(n) if _is_empty_beat(result[i])]
    if empty_indices:
        fill_count = min(num_fills, len(empty_indices))
        fill_indices = random.sample(empty_indices, fill_count)
        for i in fill_indices:
            new_notes = _generate_random_beat()
            new_tokens = encode_beat(new_notes)
            result[i]['tokens'] = new_tokens
            changed_mask[i] = True

    # Whole-beat deletion on some non-empty beats
    delete_ratio = random.uniform(0.12, 0.30)
    num_deletes = max(1, int(n * delete_ratio))
    non_empty_indices = [i for i in range(n) if not _is_empty_beat(result[i])]
    if non_empty_indices:
        delete_count = min(num_deletes, len(non_empty_indices))
        delete_indices = random.sample(non_empty_indices, delete_count)
        for i in delete_indices:
            result[i] = _make_empty_beat(result[i])
            changed_mask[i] = True

    # Simplification on some non-empty beats
    simplify_ratio = random.uniform(0.08, 0.18)
    num_simplify = max(1, int(n * simplify_ratio))
    simplify_indices = random.sample(range(n), min(num_simplify, n))

    for i in simplify_indices:
        notes = decode_beat(result[i]['tokens'])
        if len(notes) > 1:
            simplified = _simplify_beat(notes)
            new_tokens = encode_beat(simplified)
            result[i]['tokens'] = new_tokens
            changed_mask[i] = True

    return result, changed_mask


def _perturb_level_4(accomp_beats):
    """
    Level 4: Complete wipe (100% of beats cleared).
    All accompaniment beats become empty.
    """
    n = len(accomp_beats)
    result = []
    changed_mask = []

    for beat in accomp_beats:
        has_content = not _is_empty_beat(beat)
        result.append(_make_empty_beat(beat))
        changed_mask.append(has_content)

    return result, changed_mask


# ==================== Main Entry Point ====================

_LEVEL_FUNCTIONS = {
    1: _perturb_level_1,
    2: _perturb_level_2,
    3: _perturb_level_3,
    4: _perturb_level_4,
}


def perturb_accompaniment(accomp_beats, level_weights=(30, 30, 25, 15)):
    """
    Apply FELIX perturbation to accompaniment beats.

    Only modifies Track 1 (accompaniment). Melody (Track 0) is untouched.

    Args:
        accomp_beats: list of beat dicts for accompaniment track
        level_weights: weights for levels 1-4 (default: 30,30,25,15)

    Returns:
        (perturbed_beats, level, changed_mask) where:
        - perturbed_beats: list of perturbed beat dicts
        - level: perturbation level applied (1-4)
        - changed_mask: list of booleans indicating which beats were changed
    """
    # Sample level
    levels = [1, 2, 3, 4]
    total = sum(level_weights)
    probs = [w / total for w in level_weights]
    level = random.choices(levels, weights=probs, k=1)[0]

    perturb_fn = _LEVEL_FUNCTIONS[level]
    perturbed_beats, changed_mask = perturb_fn(accomp_beats)

    return perturbed_beats, level, changed_mask
