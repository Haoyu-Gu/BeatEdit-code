"""
Sequence Parser for FELIX-Music (Scheme D: absolute_bundled encoding).

Handles parsing token sequences into structured beat representations,
encoding/decoding between absolute pitch space and bundled tokens,
and track separation/reconstruction for melody/accompaniment.

Key difference from with_pair (Scheme C):
- Each bundled token uses absolute position: bundled = abs_pitch × 81 + val
- No relative position accumulation needed in encode/decode
- Each token is independently decodable

Key features beyond base parser:
- separate_tracks(): split parsed beats into melody (T0) and accompaniment (T1)
- get_beat_token_count(): count bundled tokens in a beat
- rebuild_interleaved(): reconstruct full sequence from separated tracks
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

    Uses absolute position encoding: abs_pitch = token // 81, val = token % 81.
    Each token is independently decodable (no state accumulation).

    Args:
        tokens: beat token list (bundled tokens only, no SPLIT marker).

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
        Empty list for empty beats.
    """
    if len(tokens) == 0:
        return []

    if len(tokens) == 1 and tokens[0] == EMPTY_MARKER:
        return []

    notes = []

    for t in tokens:
        if not is_bundled_token(t):
            break
        abs_pitch = t // PATTERN_NUM
        val = t % PATTERN_NUM
        notes.append((abs_pitch, val))

    return notes


def encode_beat(notes):
    """
    Encode (abs_pitch, patch_value) list into bundled token sequence.

    Uses absolute position encoding: bundled = abs_pitch × 81 + val.
    Each token is independently encoded (no delta computation).

    Args:
        notes: list of (abs_pitch, patch_value) tuples

    Returns:
        Token list (bundled tokens only, no SPLIT marker).
        Empty beat returns empty list [].
    """
    if len(notes) == 0:
        return []

    # Sort by pitch ascending
    notes = sorted(notes, key=lambda x: x[0])

    tokens = []
    for abs_pitch, val in notes:
        assert 0 <= abs_pitch <= MAX_PITCH, \
            f"Invalid absolute pitch: {abs_pitch}"
        assert 0 <= val < PATTERN_NUM, \
            f"Invalid patch value: {val}"
        bundled = abs_pitch * PATTERN_NUM + val
        tokens.append(bundled)

    return tokens


def parse_sequence(tokens):
    """
    Parse a complete token sequence into structured beat information.

    Args:
        tokens: full token sequence (list or tuple of ints)

    Returns:
        dict with keys:
            'header_tokens': list of header tokens [BOS?, TIME_SIG, BPM]
            'bars': list of bar dicts, each with:
                'bar_token_idx': index of BAR token in original sequence
                'beats': list of beat dicts, each with:
                    'tokens': beat content tokens (bundled only, no SPLIT)
                    'split_id': SPLIT_0 or SPLIT_1 or None (for EMPTY)
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
            i += 1  # skip SPLIT marker

            # Collect bundled tokens until next control token
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
            # Skip unexpected token
            i += 1

    return result


def reassemble_sequence(beats_info, new_beat_tokens_list):
    """
    Reassemble a full token sequence from parsed structure and new beat tokens.

    Args:
        beats_info: output from parse_sequence() (original structure)
        new_beat_tokens_list: list of (split_id_or_None, bundled_tokens) tuples,
                             same order as beats_info['beats'].
                             Or list of bundled token lists (split_id from original).

    Returns:
        New token sequence as list of ints.
    """
    result = list(beats_info['header_tokens'])

    beat_idx = 0
    for bar in beats_info['bars']:
        result.append(BAR_TOKEN)
        for original_beat in bar['beats']:
            if beat_idx < len(new_beat_tokens_list):
                new_tokens = new_beat_tokens_list[beat_idx]

                if isinstance(new_tokens, tuple) and len(new_tokens) == 2:
                    split_id, bundled = new_tokens
                else:
                    # Use original split_id
                    split_id = original_beat.get('split_id', None)
                    bundled = new_tokens

                if len(bundled) == 0:
                    # Empty beat
                    result.append(EMPTY_MARKER)
                else:
                    if split_id is not None:
                        result.append(split_id)
                    result.extend(bundled)
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
    Count the number of bundled tokens in a beat.

    Args:
        beat: beat dict from parse_sequence() with 'tokens' key

    Returns:
        int: number of bundled tokens (0 for empty beats)
    """
    return len(beat['tokens'])


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
            if len(beat['tokens']) == 0:
                result.append(EMPTY_MARKER)
            else:
                split_id = beat.get('split_id', None)
                if split_id is not None:
                    result.append(split_id)
                result.extend(beat['tokens'])

    result.extend(parsed_info['footer_tokens'])
    return result
