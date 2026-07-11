"""
Sequence Parser for CPWord encoding.

Parses CPWord compound token sequences into structured bar/note representations.
Simpler than REMI because each note is exactly 1 compound token.

CPWord structure per bar:
    Bar_token [Position_token Note_token+]+ ...
"""

from config import (
    FAMILY_METRIC, FAMILY_NOTE,
    POS_BAR, POS_OFFSET, POS_IGNORE,
    PITCH_OFFSET, PITCH_IGNORE, PITCH_MIN_ID, PITCH_MAX_ID,
    VEL_IGNORE, VEL_OFFSET, VEL_MAX_ID,
    DUR_IGNORE, DUR_OFFSET, DUR_MAX_ID,
    MIDI_PITCH_MIN,
    is_bar_token, is_position_token, is_note_token,
    is_special_token, is_eos_token,
    get_position_value,
)


def parse_sequence(tokens):
    """
    Parse a complete CPWord compound token sequence into structured bar/note info.

    Args:
        tokens: list of compound tokens (each is [family, pos, pitch, vel, dur])

    Returns:
        dict with keys:
            'header_tokens': list of header compound tokens (BOS etc.)
            'bars': list of bar dicts, each with:
                'bar_idx': index in tokens list
                'positions': list of position dicts, each with:
                    'position_value': 0-31
                    'position_idx': index in tokens list
                    'notes': list of note dicts, each with:
                        'token_idx': index in tokens list
                        'pitch_id': pitch sub-token ID
                        'velocity_id': velocity sub-token ID
                        'duration_id': duration sub-token ID
                        'midi_pitch': MIDI pitch value
            'footer_tokens': list of footer tokens (EOS etc.)
    """
    n = len(tokens)
    result = {
        'header_tokens': [],
        'bars': [],
        'footer_tokens': [],
    }

    i = 0

    # Parse header (BOS, any special tokens before first Bar)
    while i < n:
        if is_bar_token(tokens[i]):
            break
        result['header_tokens'].append(list(tokens[i]))
        i += 1

    # Parse bars
    while i < n:
        tok = tokens[i]

        if is_eos_token(tok):
            result['footer_tokens'].append(list(tok))
            i += 1
            break

        if is_bar_token(tok):
            bar = {
                'bar_idx': i,
                'positions': [],
            }
            result['bars'].append(bar)
            i += 1

            # Parse positions within this bar
            while i < n and not is_bar_token(tokens[i]) and not is_eos_token(tokens[i]):
                if is_position_token(tokens[i]):
                    pos_dict = {
                        'position_value': get_position_value(tokens[i]),
                        'position_idx': i,
                        'notes': [],
                    }
                    i += 1

                    # Parse notes at this position
                    while i < n and is_note_token(tokens[i]):
                        note = {
                            'token_idx': i,
                            'pitch_id': tokens[i][2],
                            'velocity_id': tokens[i][3],
                            'duration_id': tokens[i][4],
                            'midi_pitch': tokens[i][2] - PITCH_OFFSET + MIDI_PITCH_MIN,
                        }
                        pos_dict['notes'].append(note)
                        i += 1

                    bar['positions'].append(pos_dict)
                elif is_note_token(tokens[i]):
                    # Note without preceding Position (shouldn't happen normally)
                    i += 1
                else:
                    i += 1
        else:
            i += 1

    # Remaining tokens as footer
    while i < n:
        result['footer_tokens'].append(list(tokens[i]))
        i += 1

    return result


def bars_to_flat_notes(parsed):
    """
    Convert parsed structure to flat list of note tuples.

    Returns:
        list of tuples: (bar_idx, position_value, midi_pitch, velocity_id, duration_id)
    """
    notes = []
    for bar_i, bar in enumerate(parsed['bars']):
        for pos in bar['positions']:
            for note in pos['notes']:
                notes.append((
                    bar_i,
                    pos['position_value'],
                    note['midi_pitch'],
                    note['velocity_id'],
                    note['duration_id'],
                ))
    return notes


def flat_notes_to_bars(notes_list, num_bars):
    """
    Convert flat note tuples back to bar data for reassembly.

    Args:
        notes_list: list of (bar_idx, position_value, midi_pitch, velocity_id, duration_id)
        num_bars: total number of bars

    Returns:
        list of bar data for reassemble_sequence()
    """
    bars = [[] for _ in range(num_bars)]

    # Group by bar and position
    bar_pos = {}
    for bar_idx, pos_val, midi_pitch, vel_id, dur_id in notes_list:
        key = (bar_idx, pos_val)
        if key not in bar_pos:
            bar_pos[key] = []
        pitch_id = midi_pitch - MIDI_PITCH_MIN + PITCH_OFFSET
        bar_pos[key].append({
            'pitch_id': pitch_id,
            'velocity_id': vel_id,
            'duration_id': dur_id,
        })

    for (bar_idx, pos_val), note_list in sorted(bar_pos.items()):
        if 0 <= bar_idx < num_bars:
            note_list.sort(key=lambda n: n['pitch_id'])
            bars[bar_idx].append({
                'position_value': pos_val,
                'notes': note_list,
            })

    return bars


def reassemble_sequence(parsed, new_bars_data):
    """
    Reassemble a CPWord compound token sequence from parsed structure and new bar data.

    Args:
        parsed: output from parse_sequence() (used for header/footer)
        new_bars_data: list of bar data, each is a list of position groups:
            [{'position_value': int, 'notes': [{'pitch_id', 'velocity_id', 'duration_id'}]}]

    Returns:
        New compound token sequence as list of lists.
    """
    result = [list(t) for t in parsed['header_tokens']]

    for bar_data in new_bars_data:
        # Bar token
        result.append([FAMILY_METRIC, POS_BAR, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE])
        for pos_group in bar_data:
            # Position token
            pos_id = POS_OFFSET + pos_group['position_value']
            result.append([FAMILY_METRIC, pos_id, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE])
            for note in pos_group['notes']:
                result.append([
                    FAMILY_NOTE,
                    POS_IGNORE,
                    note['pitch_id'],
                    note['velocity_id'],
                    note['duration_id'],
                ])

    result.extend([list(t) for t in parsed['footer_tokens']])
    return result


def get_bar_ranges(tokens):
    """
    Get the token index ranges for each bar in the compound token sequence.

    Returns:
        list of (start_idx, end_idx) tuples, one per bar
    """
    bar_starts = []
    for i, tok in enumerate(tokens):
        if is_bar_token(tok):
            bar_starts.append(i)

    ranges = []
    for j, start in enumerate(bar_starts):
        if j + 1 < len(bar_starts):
            end = bar_starts[j + 1]
        else:
            end = len(tokens)
            for k in range(start + 1, len(tokens)):
                if is_eos_token(tokens[k]):
                    end = k
                    break
        ranges.append((start, end))

    return ranges
