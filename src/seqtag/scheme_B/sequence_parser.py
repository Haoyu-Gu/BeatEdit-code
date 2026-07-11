"""
Sequence Parser for no_pair_related encoding.

Handles parsing token sequences into structured beat representations,
and encoding/decoding between absolute pitch space and relative position tokens.
"""

from config import (
    POSITION_OFFSET, POSITION_MIN, POSITION_MAX,
    PATCH_VALUE_MIN, PATCH_VALUE_MAX,
    EMPTY_MARKER, END_MARKER, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, NUM_TIME_SIGS, BPM_OFFSET, NUM_BPMS, MAX_PITCH,
    is_position_token, is_patch_value, is_control_token, is_header_token,
)


def decode_beat(tokens):
    """
    Decode a beat's token sequence into (abs_pitch, patch_value) list.

    Args:
        tokens: beat token list, e.g. [84, 66, 88, 40, 149, 7, 170] or [169]

    Returns:
        List of (abs_pitch, patch_value) tuples, sorted by pitch ascending.
        Empty list for empty beats.

    Examples:
        >>> decode_beat([84, 66, 88, 40, 149, 7, 170])
        [(3, 66), (10, 40), (78, 7)]
        >>> decode_beat([169])
        []
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
            # Malformed: position token without valid value
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

    Examples:
        >>> encode_beat([(3, 66), (10, 40), (78, 7)])
        [84, 66, 88, 40, 149, 7, 170]
        >>> encode_beat([])
        [169]
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

    Args:
        tokens: full token sequence (list or tuple of ints)

    Returns:
        dict with keys:
            'header_tokens': list of header tokens [BOS?, TIME_SIG, BPM]
            'bars': list of bar dicts, each with:
                'bar_token_idx': index of BAR token in original sequence
                'beats': list of beat dicts, each with:
                    'tokens': beat content tokens
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


def parse_note_pairs(beat_tokens):
    """
    Parse beat tokens into note-level pairs with absolute pitch info.

    Args:
        beat_tokens: token list for one beat

    Returns:
        List of (pos_token, val_token, abs_pitch) tuples.
        Empty list for empty beats.
    """
    if len(beat_tokens) == 1 and beat_tokens[0] == EMPTY_MARKER:
        return []

    pairs = []
    abs_pitch = 0
    i = 0

    while i < len(beat_tokens):
        tok = beat_tokens[i]

        if tok == END_MARKER:
            break

        if is_position_token(tok) and i + 1 < len(beat_tokens):
            val_tok = beat_tokens[i + 1]
            if is_patch_value(val_tok):
                rel_pos = tok - POSITION_OFFSET
                abs_pitch += rel_pos
                pairs.append((tok, val_tok, abs_pitch))
                i += 2
                continue

        i += 1

    return pairs
