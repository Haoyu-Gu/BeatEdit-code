"""
Sequence Parser for REMI encoding.

Parses REMI token sequences into structured bar/note representations.
Handles the key REMI property: notes at the same Position share one Position token.

REMI structure per bar:
    Bar [Position [Pitch Velocity Duration]+]+ ...

So within a bar, the sequence is:
    Position_X Pitch_A Vel_A Dur_A Pitch_B Vel_B Dur_B Position_Y Pitch_C Vel_C Dur_C ...
"""

from config import (
    BAR_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN,
    PITCH_OFFSET, MIDI_PITCH_MIN,
    VELOCITY_OFFSET, DURATION_OFFSET, POSITION_OFFSET,
    is_pitch_token, is_velocity_token, is_duration_token,
    is_position_token, is_music_token, is_special_token,
)


def parse_sequence(tokens):
    """
    Parse a complete REMI token sequence into structured bar/note information.

    Args:
        tokens: full token sequence (list of ints)

    Returns:
        dict with keys:
            'header_tokens': list of header tokens before first Bar
            'bars': list of bar dicts, each with:
                'bar_token_idx': index of BAR token in original sequence
                'positions': list of position dicts, each with:
                    'position_token': Position token ID
                    'position_value': position value (0-31)
                    'position_token_idx': index in original sequence
                    'notes': list of note dicts, each with:
                        'pitch_token': Pitch token ID
                        'velocity_token': Velocity token ID
                        'duration_token': Duration token ID
                        'start_idx': start index in original sequence
                        'end_idx': end index (exclusive) in original sequence
                'start_idx': start index of this position group
                'end_idx': end index (exclusive)
            'footer_tokens': list of footer tokens (EOS etc.)
            'all_notes': flat list of all note dicts
    """
    tokens = list(tokens)
    n = len(tokens)

    result = {
        'header_tokens': [],
        'bars': [],
        'footer_tokens': [],
        'all_notes': [],
    }

    i = 0

    # Parse header (BOS, any special tokens before first Bar)
    while i < n:
        t = tokens[i]
        if t == BAR_TOKEN:
            break
        if t in (BOS_TOKEN, PAD_TOKEN):
            result['header_tokens'].append(t)
            i += 1
        else:
            # Non-standard header token, include it
            result['header_tokens'].append(t)
            i += 1

    # Parse bars
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
                'positions': [],
            }
            result['bars'].append(bar)
            i += 1

            # Parse positions within this bar
            while i < n and tokens[i] != BAR_TOKEN and tokens[i] != EOS_TOKEN and tokens[i] != PAD_TOKEN:
                if is_position_token(tokens[i]):
                    pos_dict = {
                        'position_token': tokens[i],
                        'position_value': tokens[i] - POSITION_OFFSET,
                        'position_token_idx': i,
                        'notes': [],
                        'start_idx': i,
                        'end_idx': i + 1,
                    }
                    i += 1

                    # Parse notes at this position: [Pitch Vel Dur]+
                    while i < n and is_pitch_token(tokens[i]):
                        note_start = i
                        pitch_tok = tokens[i]
                        i += 1

                        vel_tok = tokens[i] if i < n and is_velocity_token(tokens[i]) else None
                        if vel_tok is not None:
                            i += 1

                        dur_tok = tokens[i] if i < n and is_duration_token(tokens[i]) else None
                        if dur_tok is not None:
                            i += 1

                        if vel_tok is not None and dur_tok is not None:
                            note = {
                                'pitch_token': pitch_tok,
                                'velocity_token': vel_tok,
                                'duration_token': dur_tok,
                                'midi_pitch': pitch_tok - PITCH_OFFSET + MIDI_PITCH_MIN,
                                'start_idx': note_start,
                                'end_idx': i,
                            }
                            pos_dict['notes'].append(note)
                            result['all_notes'].append(note)

                    pos_dict['end_idx'] = i
                    bar['positions'].append(pos_dict)
                else:
                    # Skip unexpected tokens
                    i += 1
        else:
            i += 1

    # Collect remaining tokens as footer
    while i < n:
        result['footer_tokens'].append(tokens[i])
        i += 1

    return result


def reassemble_sequence(parsed, new_bars_data):
    """
    Reassemble a REMI token sequence from parsed structure and new bar data.

    Args:
        parsed: output from parse_sequence() (used for header/footer)
        new_bars_data: list of bar data, each is a list of position groups:
            [{'position_token': int, 'notes': [{'pitch_token', 'velocity_token', 'duration_token'}]}]

    Returns:
        New token sequence as list of ints.
    """
    result = list(parsed['header_tokens'])

    for bar_data in new_bars_data:
        result.append(BAR_TOKEN)
        for pos_group in bar_data:
            result.append(pos_group['position_token'])
            for note in pos_group['notes']:
                result.append(note['pitch_token'])
                result.append(note['velocity_token'])
                result.append(note['duration_token'])

    result.extend(parsed['footer_tokens'])
    return result


def bars_to_flat_notes(parsed):
    """
    Convert parsed structure to flat list of (bar_idx, position, pitch, vel, dur) tuples.

    Args:
        parsed: output from parse_sequence()

    Returns:
        list of tuples: (bar_idx, position_value, midi_pitch, velocity_token, duration_token)
    """
    notes = []
    for bar_idx, bar in enumerate(parsed['bars']):
        for pos in bar['positions']:
            for note in pos['notes']:
                notes.append((
                    bar_idx,
                    pos['position_value'],
                    note['midi_pitch'],
                    note['velocity_token'],
                    note['duration_token'],
                ))
    return notes


def flat_notes_to_bars(notes_list, num_bars):
    """
    Convert flat note tuples back to bar data structure for reassembly.

    Args:
        notes_list: list of (bar_idx, position_value, midi_pitch, velocity_token, duration_token)
        num_bars: total number of bars

    Returns:
        list of bar data for reassemble_sequence()
    """
    bars = [[] for _ in range(num_bars)]

    # Group by bar and position
    bar_pos = {}
    for bar_idx, pos_val, midi_pitch, vel_tok, dur_tok in notes_list:
        key = (bar_idx, pos_val)
        if key not in bar_pos:
            bar_pos[key] = []
        pitch_tok = midi_pitch - MIDI_PITCH_MIN + PITCH_OFFSET
        bar_pos[key].append({
            'pitch_token': pitch_tok,
            'velocity_token': vel_tok,
            'duration_token': dur_tok,
        })

    for (bar_idx, pos_val), note_list in sorted(bar_pos.items()):
        if 0 <= bar_idx < num_bars:
            # Sort notes by pitch within position
            note_list.sort(key=lambda n: n['pitch_token'])
            bars[bar_idx].append({
                'position_token': POSITION_OFFSET + pos_val,
                'notes': note_list,
            })

    return bars


def get_bar_token_ranges(tokens):
    """
    Get the token index ranges for each bar in the sequence.

    Args:
        tokens: REMI token sequence

    Returns:
        list of (start_idx, end_idx) tuples, one per bar
    """
    bar_starts = []
    for i, tok in enumerate(tokens):
        if tok == BAR_TOKEN:
            bar_starts.append(i)

    ranges = []
    for j, start in enumerate(bar_starts):
        if j + 1 < len(bar_starts):
            end = bar_starts[j + 1]
        else:
            # Last bar: extends to EOS or end of sequence
            end = len(tokens)
            for k in range(start + 1, len(tokens)):
                if tokens[k] == EOS_TOKEN:
                    end = k
                    break
        ranges.append((start, end))

    return ranges
