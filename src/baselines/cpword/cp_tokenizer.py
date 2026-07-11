"""
CPWord Tokenizer Wrapper

Wraps miditok CPWord tokenizer for consistent interface with the GECToR pipeline.
Handles MIDI -> compound tokens and compound tokens -> note list conversions.

Each compound token = [family_id, position_id, pitch_id, velocity_id, duration_id].
"""

from miditok import CPWord, TokenizerConfig
from symusic import Score

from config import (
    FAMILY_METRIC, FAMILY_NOTE, FAMILY_BOS, FAMILY_EOS,
    POS_BAR, POS_IGNORE, POS_OFFSET, POS_MAX,
    PITCH_IGNORE, PITCH_OFFSET, MIDI_PITCH_MIN, MIDI_PITCH_MAX,
    PITCH_MIN_ID, PITCH_MAX_ID,
    VEL_IGNORE, VEL_OFFSET, VEL_MAX_ID,
    DUR_IGNORE, DUR_OFFSET, DUR_MAX_ID,
    BOS_TOKEN, EOS_TOKEN,
    is_bar_token, is_position_token, is_note_token,
    is_special_token, get_position_value, get_midi_pitch,
)

# Singleton tokenizer instance
_tokenizer = None


def create_cp_tokenizer():
    """Create a configured miditok CPWord tokenizer."""
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=False,
    )
    return CPWord(config)


def get_tokenizer():
    """Get or create singleton tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = create_cp_tokenizer()
    return _tokenizer


def midi_to_compound_tokens(midi_path):
    """
    Tokenize a MIDI file to CPWord compound tokens.

    Returns a list of compound tokens, each is [family, position, pitch, velocity, duration].
    """
    tokenizer = get_tokenizer()
    score = Score(midi_path)
    tok_sequences = tokenizer(score)

    if len(tok_sequences) == 0:
        return [[FAMILY_METRIC, POS_BAR, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE]]

    ids = tok_sequences[0].ids
    return [list(t) for t in ids]


def compound_tokens_to_notes(tokens):
    """
    Parse compound tokens into a structured note list.

    Args:
        tokens: list of compound tokens (each is [family, pos, pitch, vel, dur])

    Returns:
        list of dicts, each with:
            'bar': int (0-indexed bar number)
            'position': int (position value, 0-31)
            'pitch': int (MIDI pitch, 21-109)
            'velocity_id': int (velocity sub-token ID)
            'duration_id': int (duration sub-token ID)
    """
    notes = []
    current_bar = -1
    current_position = None

    for tok in tokens:
        if is_special_token(tok):
            continue
        if is_bar_token(tok):
            current_bar += 1
            current_position = None
        elif is_position_token(tok):
            current_position = get_position_value(tok)
        elif is_note_token(tok):
            pitch_id = tok[2]
            vel_id = tok[3]
            dur_id = tok[4]
            if PITCH_MIN_ID <= pitch_id <= PITCH_MAX_ID and \
               VEL_OFFSET <= vel_id <= VEL_MAX_ID and \
               DUR_OFFSET <= dur_id <= DUR_MAX_ID:
                midi_pitch = pitch_id - PITCH_OFFSET + MIDI_PITCH_MIN
                notes.append({
                    'bar': current_bar,
                    'position': current_position if current_position is not None else 0,
                    'pitch': midi_pitch,
                    'velocity_id': vel_id,
                    'duration_id': dur_id,
                })

    return notes


def notes_to_compound_tokens(notes, num_bars=None):
    """
    Convert structured note list back to compound token sequence.

    Args:
        notes: list of note dicts (from compound_tokens_to_notes)
        num_bars: total number of bars (if None, inferred from notes)

    Returns:
        list of compound tokens
    """
    if not notes:
        return [[FAMILY_METRIC, POS_BAR, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE]]

    if num_bars is None:
        num_bars = max(n['bar'] for n in notes) + 1

    # Group notes by bar and position
    bars = {}
    for note in notes:
        bar_idx = note['bar']
        if bar_idx not in bars:
            bars[bar_idx] = {}
        pos = note['position']
        if pos not in bars[bar_idx]:
            bars[bar_idx][pos] = []
        bars[bar_idx][pos].append(note)

    tokens = []
    for bar_idx in range(num_bars):
        # Bar token
        tokens.append([FAMILY_METRIC, POS_BAR, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE])
        if bar_idx not in bars:
            continue
        for pos in sorted(bars[bar_idx].keys()):
            # Position token
            pos_id = POS_OFFSET + pos
            if pos_id > POS_MAX:
                pos_id = POS_MAX
            tokens.append([FAMILY_METRIC, pos_id, PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE])
            # Notes sorted by pitch
            pos_notes = sorted(bars[bar_idx][pos], key=lambda n: n['pitch'])
            for note in pos_notes:
                pitch_id = note['pitch'] - MIDI_PITCH_MIN + PITCH_OFFSET
                tokens.append([
                    FAMILY_NOTE,
                    POS_IGNORE,
                    pitch_id,
                    note['velocity_id'],
                    note['duration_id'],
                ])

    return tokens


def get_sub_vocab_sizes():
    """Get the sub-vocabulary sizes from miditok."""
    tokenizer = get_tokenizer()
    return tuple(len(v) for v in tokenizer.vocab)
