"""
Sequence Parser for with_pair encoding.

Handles parsing token sequences into structured beat representations,
and encoding/decoding between absolute pitch space and bundled tokens.

Key difference from no_pair_related:
- Each note is a single bundled token (relative_pos × 81 + patch_value)
- Beats start with SPLIT_0/SPLIT_1, no END_MARKER
- Beat boundary determined by next control token (>=7128)
"""

from config import (
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
                e.g. [246, 567, 1200] or [] for empty beats.

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
        Empty list for empty beats.

    Examples:
        >>> decode_beat([246])  # 246 = 3*81 + 3 → pitch=3, val=3
        [(3, 3)]
        >>> decode_beat([])
        []
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
        Empty beat returns empty list [].

    Examples:
        >>> encode_beat([(3, 3), (10, 40)])
        [246, 607]  # 3*81+3=246, 7*81+40=607
        >>> encode_beat([])
        []
    """
    if len(notes) == 0:
        return []

    # Sort by pitch ascending
    notes = sorted(notes, key=lambda x: x[0])

    tokens = []
    prev_pitch = 0
    for abs_pitch, val in notes:
        rel_pos = abs_pitch - prev_pitch
        assert 0 <= rel_pos <= MAX_PITCH, \
            f"Invalid relative position: {rel_pos} (pitch {abs_pitch}, prev {prev_pitch})"
        assert 0 <= val < PATTERN_NUM, \
            f"Invalid patch value: {val}"
        bundled = rel_pos * PATTERN_NUM + val
        tokens.append(bundled)
        prev_pitch = abs_pitch

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
