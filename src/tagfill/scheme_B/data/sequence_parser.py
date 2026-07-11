"""
Sequence Parser for FELIX-Music (Scheme B: no_pair_related encoding).

Handles parsing token sequences into structured beat representations,
encoding/decoding between absolute pitch space and 2-token-per-note format,
and track separation/reconstruction for melody/accompaniment.

Key encoding format:
- Each note = [position_token (81+rel_distance)][value_token (0-80)]
- Non-empty beat = [pos][val][pos][val]...[END_MARKER(170)]
- Empty beat = [EMPTY_MARKER(169)]
- No SPLIT markers; track interleaving: T0_B0, T1_B0, T0_B1, T1_B1, ...

Key features:
- separate_tracks(): split parsed beats into melody (T0) and accompaniment (T1)
- get_beat_token_count(): count music tokens in a beat
- rebuild_interleaved(): reconstruct full sequence from separated tracks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from configs.config import (
    POSITION_OFFSET, POSITION_MIN, POSITION_MAX,
    PATCH_VALUE_MIN, PATCH_VALUE_MAX, PATTERN_NUM, MAX_PITCH,
    EMPTY_MARKER, END_MARKER, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, NUM_TIME_SIGS, BPM_OFFSET, NUM_BPMS,
    is_position_token, is_patch_value, is_control_token, is_header_token,
)


def decode_beat(tokens):
    """
    Decode a beat's token sequence into (abs_pitch, patch_value) list.

    In Scheme B, each note is [position_token][value_token] with relative positions.
    Position accumulates: abs_pitch += (pos_token - 81).

    Args:
        tokens: beat token list (including END_MARKER if non-empty, or
                [EMPTY_MARKER] for empty beats).

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
        Empty list for empty beats.
    """
    if len(tokens) == 0:
        return []

    if len(tokens) == 1 and tokens[0] == EMPTY_MARKER:
        return []

    notes = []
    abs_pitch = 0
    i = 0

    while i < len(tokens):
        if tokens[i] == END_MARKER:
            break

        pos_token = tokens[i]
        if not is_position_token(pos_token):
            break

        i += 1
        if i >= len(tokens):
            break

        val_token = tokens[i]
        if not is_patch_value(val_token):
            break

        rel_pos = pos_token - POSITION_OFFSET
        abs_pitch = abs_pitch + rel_pos
        notes.append((abs_pitch, val_token))
        i += 1

    return notes


def encode_beat(notes):
    """
    Encode (abs_pitch, patch_value) list into token sequence.

    Args:
        notes: list of (abs_pitch, patch_value) tuples

    Returns:
        Token list. Empty beat returns [EMPTY_MARKER].
        Non-empty beat returns [pos, val, pos, val, ..., END_MARKER].
    """
    if len(notes) == 0:
        return [EMPTY_MARKER]

    # Sort by pitch ascending
    notes = sorted(notes, key=lambda x: x[0])

    tokens = []
    prev_pitch = 0
    for abs_pitch, val in notes:
        rel_pos = abs_pitch - prev_pitch
        assert 0 <= rel_pos <= MAX_PITCH, \
            f"Invalid relative position: {rel_pos} (pitch {abs_pitch}, prev {prev_pitch})"
        assert PATCH_VALUE_MIN <= val <= PATCH_VALUE_MAX, \
            f"Invalid patch value: {val}"
        tokens.append(POSITION_OFFSET + rel_pos)
        tokens.append(val)
        prev_pitch = abs_pitch

    tokens.append(END_MARKER)
    return tokens


def parse_sequence(tokens):
    """
    Parse a complete token sequence into structured beat information.

    In Scheme B, beats are delimited by:
    - EMPTY_MARKER (169) for empty beats
    - Position token (81-168) starting a non-empty beat, ending at END_MARKER (170)

    Args:
        tokens: full token sequence (list or tuple of ints)

    Returns:
        dict with keys:
            'header_tokens': list of header tokens [BOS?, TIME_SIG, BPM]
            'bars': list of bar dicts, each with:
                'bar_token_idx': index of BAR token in original sequence
                'beats': list of beat dicts, each with:
                    'tokens': beat content tokens (including END_MARKER for non-empty,
                              or [EMPTY_MARKER] for empty)
                    'start_idx': start position in original sequence
                    'end_idx': end position in original sequence (exclusive)
            'footer_tokens': list of footer tokens [EOS?]
            'beats': flat list of all beat dicts in sequence order
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

    # Parse header (BOS?, TIME_SIG, BPM)
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
            bar = {
                'bar_token_idx': i,
                'beats': [],
            }
            result['bars'].append(bar)
            i += 1
            continue

        # No bars seen yet - skip unexpected tokens
        if len(result['bars']) == 0:
            i += 1
            continue

        current_bar = result['bars'][-1]

        # Parse a beat
        if t == EMPTY_MARKER:
            beat = {
                'tokens': [EMPTY_MARKER],
                'start_idx': i,
                'end_idx': i + 1,
            }
            current_bar['beats'].append(beat)
            result['beats'].append(beat)
            i += 1

        elif is_position_token(t):
            # Non-empty beat: scan until END_MARKER or next structural token
            start_idx = i
            beat_tokens = []

            while i < n:
                tok = tokens[i]

                if tok == END_MARKER:
                    beat_tokens.append(END_MARKER)
                    i += 1
                    break

                # Stop at structural boundaries
                if tok == BAR_TOKEN or tok == EOS_TOKEN or tok == EMPTY_MARKER:
                    break

                # Accept position and value tokens
                if is_position_token(tok) or is_patch_value(tok):
                    beat_tokens.append(tok)
                    i += 1
                else:
                    # Unexpected token - stop beat
                    break

            beat = {
                'tokens': beat_tokens,
                'start_idx': start_idx,
                'end_idx': i,
            }
            current_bar['beats'].append(beat)
            result['beats'].append(beat)

        else:
            # Skip unexpected token
            i += 1

    return result


def reassemble_sequence(beats_info, new_beat_tokens_list):
    """
    Reassemble a full token sequence from parsed structure and new beat tokens.

    Args:
        beats_info: output from parse_sequence() (original structure)
        new_beat_tokens_list: list of new token lists for each beat,
                             same order as beats_info['beats']

    Returns:
        New token sequence as list of ints.
    """
    result = list(beats_info['header_tokens'])

    beat_idx = 0
    for bar in beats_info['bars']:
        result.append(BAR_TOKEN)
        for _ in bar['beats']:
            if beat_idx < len(new_beat_tokens_list):
                result.extend(new_beat_tokens_list[beat_idx])
            beat_idx += 1

    result.extend(beats_info['footer_tokens'])
    return result


# ==================== FELIX-specific Track Separation ====================

def separate_tracks(parsed_info):
    """
    Separate parsed beats into melody (Track 0) and accompaniment (Track 1).

    Within each bar, beats are interleaved: T0_B0, T1_B0, T0_B1, T1_B1, ...
    So beat_index % 2 == 0 is Track 0 (melody), == 1 is Track 1 (accompaniment).

    Args:
        parsed_info: output from parse_sequence()

    Returns:
        (melody_beats, accomp_beats) where each is a list of beat dicts.
        Both lists have the same length (one entry per musical beat position).
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


def get_beat_token_count(beat):
    """
    Count the number of tokens in a beat (including END_MARKER if present).

    Args:
        beat: beat dict from parse_sequence() with 'tokens' key

    Returns:
        int: number of tokens (0 for empty beats represented as [EMPTY_MARKER])
    """
    tokens = beat['tokens']
    if len(tokens) == 1 and tokens[0] == EMPTY_MARKER:
        return 0
    return len(tokens)


def rebuild_interleaved(melody_beats, accomp_beats, parsed_info):
    """
    Reconstruct a full token sequence from separated melody and accompaniment beats.

    Re-interleaves melody (T0) and accompaniment (T1) beats within each bar,
    preserving the original header/footer and bar structure.

    Args:
        melody_beats: list of beat dicts for Track 0 (melody)
        accomp_beats: list of beat dicts for Track 1 (accompaniment)
        parsed_info: original parse_sequence() output (for structure)

    Returns:
        Token sequence as list of ints.
    """
    result = list(parsed_info['header_tokens'])

    mel_idx = 0
    acc_idx = 0

    for bar in parsed_info['bars']:
        result.append(BAR_TOKEN)
        num_bar_beats = len(bar['beats'])

        for beat_idx in range(num_bar_beats):
            if beat_idx % 2 == 0:
                # Track 0 (melody)
                if mel_idx < len(melody_beats):
                    beat = melody_beats[mel_idx]
                    mel_idx += 1
                else:
                    result.append(EMPTY_MARKER)
                    continue
            else:
                # Track 1 (accompaniment)
                if acc_idx < len(accomp_beats):
                    beat = accomp_beats[acc_idx]
                    acc_idx += 1
                else:
                    result.append(EMPTY_MARKER)
                    continue

            # Emit beat tokens
            result.extend(beat['tokens'])

    result.extend(parsed_info['footer_tokens'])
    return result
