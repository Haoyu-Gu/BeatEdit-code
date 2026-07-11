"""
Sequence Parser for FELIX-Music (Scheme A: no_pair encoding).

Handles parsing token sequences into structured beat representations,
encoding/decoding between absolute pitch space and 2-token-per-note format,
and track separation/reconstruction for melody/accompaniment.

Key encoding format:
- Each note = [abs_position_token (81+pitch)][value_token (0-80)]
- Non-empty beat = [TRACK_MARKER][pos][val][pos][val]...
- Empty beat = [TRACK_MARKER][0]
- No END_MARKER: beat boundary = next control token (>=170) or track marker

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
    TRACK0_START, TRACK1_START,
    BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, NUM_TIME_SIGS, BPM_OFFSET, NUM_BPMS,
    is_position_token, is_patch_value, is_control_token, is_track_marker,
    is_header_token,
)


def decode_beat(tokens):
    """
    Decode a beat's token sequence into (abs_pitch, patch_value) list.

    In Scheme A, each note is [abs_position][patch_value] where
    abs_position = 81 + pitch_index. Positions are absolute (no accumulation).

    Args:
        tokens: beat token list (without track marker).
                e.g. [84, 66, 88, 40] for two notes, or [0] for empty beat.

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
        Empty list for empty beats.
    """
    if len(tokens) == 0:
        return []

    # Empty beat: [0] (single zero token after track marker)
    if len(tokens) == 1 and tokens[0] == 0:
        return []

    notes = []
    i = 0

    while i < len(tokens):
        pos_token = tokens[i]
        if not is_position_token(pos_token):
            break

        i += 1
        if i >= len(tokens):
            break

        val_token = tokens[i]
        if not is_patch_value(val_token):
            break

        abs_pitch = pos_token - POSITION_OFFSET
        notes.append((abs_pitch, val_token))
        i += 1

    return notes


def encode_beat(notes):
    """
    Encode (abs_pitch, patch_value) list into token sequence.

    Uses absolute position encoding: each note becomes [81+pitch][value].
    Does NOT include track marker.

    Args:
        notes: list of (abs_pitch, patch_value) tuples

    Returns:
        Token list. Empty beat returns [0].
        Non-empty beat returns [pos, val, pos, val, ...].
    """
    if len(notes) == 0:
        return [0]

    # Sort by pitch ascending
    notes = sorted(notes, key=lambda x: x[0])

    tokens = []
    for abs_pitch, val in notes:
        assert 0 <= abs_pitch <= MAX_PITCH, \
            f"Invalid pitch: {abs_pitch}"
        assert PATCH_VALUE_MIN <= val <= PATCH_VALUE_MAX, \
            f"Invalid patch value: {val}"
        tokens.append(POSITION_OFFSET + abs_pitch)
        tokens.append(val)

    return tokens


def parse_sequence(tokens):
    """
    Parse a complete token sequence into structured beat information.

    In Scheme A, beat structure is:
    - [TRACK_MARKER][abs_pos][val][abs_pos][val]... for non-empty beats
    - [TRACK_MARKER][0] for empty beats
    Beat boundary = next control token (>=170) including track markers.

    Args:
        tokens: full token sequence (list or tuple of ints)

    Returns:
        dict with keys:
            'header_tokens': list of header tokens [BOS?, TIME_SIG, BPM]
            'bars': list of bar dicts, each with:
                'bar_token_idx': index of BAR token in original sequence
                'beats': list of beat dicts, each with:
                    'tokens': beat content tokens (WITHOUT track marker)
                    'track_id': TRACK0_START or TRACK1_START
                    'start_idx': start position in original sequence (at track marker)
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

        # Parse a beat starting with TRACK marker
        if is_track_marker(t):
            track_id = t
            start_idx = i
            i += 1  # skip track marker

            # Collect beat content tokens until next control token
            beat_tokens = []
            while i < n:
                tok = tokens[i]
                # Stop at any control token (BAR, EOS, PAD) or track marker
                if is_control_token(tok) or is_track_marker(tok):
                    break
                # Accept position and value tokens
                if is_position_token(tok) or is_patch_value(tok):
                    beat_tokens.append(tok)
                    i += 1
                else:
                    break

            beat = {
                'tokens': beat_tokens,
                'track_id': track_id,
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
        new_beat_tokens_list: list of new token lists for each beat (WITHOUT
                             track markers), same order as beats_info['beats']

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
                track_id = original_beat['track_id']
                result.append(track_id)
                result.extend(new_tokens)
            beat_idx += 1

    result.extend(beats_info['footer_tokens'])
    return result


# ==================== FELIX-specific Track Separation ====================

def separate_tracks(parsed_info):
    """
    Separate parsed beats into melody (Track 0) and accompaniment (Track 1).

    In Scheme A, track identity is explicit via TRACK0_START/TRACK1_START markers.
    Within each bar, beats alternate: T0_B0, T1_B0, T0_B1, T1_B1, ...

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
    Count the number of tokens in a beat (excluding track marker).

    Args:
        beat: beat dict from parse_sequence() with 'tokens' key

    Returns:
        int: number of content tokens (0 for empty beats [0])
    """
    tokens = beat['tokens']
    if len(tokens) == 1 and tokens[0] == 0:
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
                    result.append(TRACK0_START)
                    result.append(0)  # empty beat
                    continue
            else:
                # Track 1 (accompaniment)
                if acc_idx < len(accomp_beats):
                    beat = accomp_beats[acc_idx]
                    acc_idx += 1
                else:
                    result.append(TRACK1_START)
                    result.append(0)  # empty beat
                    continue

            # Emit track marker + beat tokens
            track_id = beat.get('track_id', TRACK0_START if beat_idx % 2 == 0 else TRACK1_START)
            result.append(track_id)
            result.extend(beat['tokens'])

    result.extend(parsed_info['footer_tokens'])
    return result
