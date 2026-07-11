"""
Sequence Parser for LevT Music Inpainting (with_pair encoding).

Reuses the parsing logic from FELIX. Handles:
- Parsing token sequences into structured beat representations
- Beat boundary detection (for mask strategies)
- Track separation (melody vs accompaniment)
- Sequence reassembly

Imported from FELIX and adapted for LevT context.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from configs.config import (
    BUNDLED_TOKEN_MIN, BUNDLED_TOKEN_MAX, PATTERN_NUM, MAX_PITCH,
    EMPTY_MARKER, SPLIT_0, SPLIT_1, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, NUM_TIME_SIGS, BPM_OFFSET, NUM_BPMS,
    is_bundled_token, is_control_token, is_split_token, is_header_token,
)


def decode_beat(tokens):
    """
    Decode a beat's bundled token sequence into (abs_pitch, patch_value) list.

    Args:
        tokens: beat token list (bundled tokens only, no SPLIT marker).

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
    """
    if len(tokens) == 0:
        return []
    if len(tokens) == 1 and tokens[0] == EMPTY_MARKER:
        return []

    notes = []
    current_pos = 0
    for t in tokens:
        if not is_bundled_token(t):
            break
        rel_pos = t // PATTERN_NUM
        val = t % PATTERN_NUM
        abs_pitch = current_pos + rel_pos
        notes.append((abs_pitch, val))
        current_pos = abs_pitch
    return notes


def encode_beat(notes):
    """
    Encode (abs_pitch, patch_value) list into bundled token sequence.

    Args:
        notes: list of (abs_pitch, patch_value) tuples

    Returns:
        Token list (bundled tokens only, no SPLIT marker).
    """
    if len(notes) == 0:
        return []

    notes = sorted(notes, key=lambda x: x[0])
    tokens = []
    prev_pitch = 0
    for abs_pitch, val in notes:
        rel_pos = abs_pitch - prev_pitch
        assert 0 <= rel_pos <= MAX_PITCH, \
            f"Invalid relative position: {rel_pos} (pitch {abs_pitch}, prev {prev_pitch})"
        assert 0 <= val < PATTERN_NUM, f"Invalid patch value: {val}"
        bundled = rel_pos * PATTERN_NUM + val
        tokens.append(bundled)
        prev_pitch = abs_pitch
    return tokens


def parse_sequence(tokens):
    """
    Parse a complete token sequence into structured beat information.

    Returns:
        dict with 'header_tokens', 'bars' (list of bar dicts with 'beats'),
        'footer_tokens', 'beats' (flat list of all beats in order).
    """
    tokens = list(tokens)
    n = len(tokens)

    result = {
        'header_tokens': [],
        'bars': [],
        'footer_tokens': [],
        'beats': [],
    }

    i = 0

    # Parse header
    while i < n:
        t = tokens[i]
        if is_header_token(t):
            result['header_tokens'].append(t)
            i += 1
        else:
            break

    # Parse bars and beats
    while i < n:
        t = tokens[i]

        if t == EOS_TOKEN:
            result['footer_tokens'].append(t)
            i += 1
            break

        if t == PAD_TOKEN:
            i += 1
            continue

        if t == BAR_TOKEN:
            bar = {'bar_token_idx': i, 'beats': []}
            result['bars'].append(bar)
            i += 1
            continue

        if len(result['bars']) == 0:
            i += 1
            continue

        current_bar = result['bars'][-1]

        if t == EMPTY_MARKER:
            beat = {
                'tokens': [],
                'split_id': None,
                'start_idx': i,
                'end_idx': i + 1,
            }
            current_bar['beats'].append(beat)
            result['beats'].append(beat)
            i += 1

        elif is_split_token(t):
            split_id = t
            start_idx = i
            i += 1

            beat_tokens = []
            while i < n:
                next_token = tokens[i]
                if is_control_token(next_token):
                    break
                if is_bundled_token(next_token):
                    beat_tokens.append(next_token)
                    i += 1
                else:
                    break

            beat = {
                'tokens': beat_tokens,
                'split_id': split_id,
                'start_idx': start_idx,
                'end_idx': i,
            }
            current_bar['beats'].append(beat)
            result['beats'].append(beat)
        else:
            i += 1

    return result


def get_beat_boundaries(tokens):
    """
    Get the token-level start indices of each beat in the sequence.

    Returns:
        List of (start_idx, end_idx) tuples for each beat.
    """
    parsed = parse_sequence(tokens)
    boundaries = []
    for beat in parsed['beats']:
        boundaries.append((beat['start_idx'], beat['end_idx']))
    return boundaries


def separate_tracks(parsed_info):
    """
    Separate parsed beats into melody (Track 0) and accompaniment (Track 1).

    Within each bar, beats are interleaved: T0_B0, T1_B0, T0_B1, T1_B1, ...
    beat_index % 2 == 0 is Track 0, == 1 is Track 1.
    """
    melody_beats = []
    accomp_beats = []
    for bar in parsed_info['bars']:
        for beat_idx, beat in enumerate(bar['beats']):
            if beat_idx % 2 == 0:
                melody_beats.append(beat)
            else:
                accomp_beats.append(beat)
    return melody_beats, accomp_beats


def reassemble_sequence(parsed_info, new_beat_tokens_list):
    """
    Reassemble a full token sequence from parsed structure and new beat tokens.

    Args:
        parsed_info: output from parse_sequence()
        new_beat_tokens_list: list of (split_id_or_None, bundled_tokens) tuples,
                             or list of bundled token lists (split_id from original).
    """
    result = list(parsed_info['header_tokens'])

    beat_idx = 0
    for bar in parsed_info['bars']:
        result.append(BAR_TOKEN)
        for original_beat in bar['beats']:
            if beat_idx < len(new_beat_tokens_list):
                new_tokens = new_beat_tokens_list[beat_idx]
                if isinstance(new_tokens, tuple) and len(new_tokens) == 2:
                    split_id, bundled = new_tokens
                else:
                    split_id = original_beat.get('split_id', None)
                    bundled = new_tokens

                if len(bundled) == 0:
                    result.append(EMPTY_MARKER)
                else:
                    if split_id is not None:
                        result.append(split_id)
                    result.extend(bundled)
            beat_idx += 1

    result.extend(parsed_info['footer_tokens'])
    return result


def rebuild_interleaved(melody_beats, accomp_beats, parsed_info):
    """
    Reconstruct a full token sequence from separated melody and accompaniment beats.
    """
    result = list(parsed_info['header_tokens'])

    mel_idx = 0
    acc_idx = 0

    for bar in parsed_info['bars']:
        result.append(BAR_TOKEN)
        num_bar_beats = len(bar['beats'])

        for beat_idx in range(num_bar_beats):
            if beat_idx % 2 == 0:
                if mel_idx < len(melody_beats):
                    beat = melody_beats[mel_idx]
                    mel_idx += 1
                else:
                    result.append(EMPTY_MARKER)
                    continue
            else:
                if acc_idx < len(accomp_beats):
                    beat = accomp_beats[acc_idx]
                    acc_idx += 1
                else:
                    result.append(EMPTY_MARKER)
                    continue

            if len(beat['tokens']) == 0:
                result.append(EMPTY_MARKER)
            else:
                split_id = beat.get('split_id', None)
                if split_id is not None:
                    result.append(split_id)
                result.extend(beat['tokens'])

    result.extend(parsed_info['footer_tokens'])
    return result
