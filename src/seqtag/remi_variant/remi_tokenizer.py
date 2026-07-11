"""
REMI Tokenizer Wrapper

Wraps miditok REMI tokenizer for consistent interface with the GECToR pipeline.
Handles MIDI → token IDs and token IDs → note list conversions.
"""

from miditok import REMI, TokenizerConfig
from symusic import Score

from config import (
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, BAR_TOKEN,
    PITCH_OFFSET, PITCH_MIN, PITCH_MAX, MIDI_PITCH_MIN,
    VELOCITY_OFFSET, VELOCITY_MIN, VELOCITY_MAX,
    DURATION_OFFSET, DURATION_MIN, DURATION_MAX,
    POSITION_OFFSET, POSITION_MIN, POSITION_MAX,
    is_pitch_token, is_velocity_token, is_duration_token, is_position_token,
)

# Singleton tokenizer instance
_tokenizer = None


def create_remi_tokenizer():
    """Create a configured miditok REMI tokenizer (no tempo/chord, 32 velocity bins, piano)."""
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=False,
    )
    return REMI(config)


def get_tokenizer():
    """Get or create singleton tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = create_remi_tokenizer()
    return _tokenizer


def midi_to_tokens(midi_path):
    """
    Tokenize a MIDI file to REMI token IDs.

    Returns a flat list of token IDs for the first track (piano).
    Multi-track MIDI: all tracks are merged into one piano track by symusic.

    Args:
        midi_path: path to MIDI file

    Returns:
        list of int token IDs
    """
    tokenizer = get_tokenizer()
    score = Score(midi_path)

    # Tokenize - returns list of TokSequence (one per track)
    tok_sequences = tokenizer(score)

    if len(tok_sequences) == 0:
        return [BAR_TOKEN]

    # Use first track
    ids = tok_sequences[0].ids
    return list(ids)


def tokens_to_events(token_ids):
    """
    Convert token IDs back to event name strings.

    Args:
        token_ids: list of int token IDs

    Returns:
        list of event name strings
    """
    tokenizer = get_tokenizer()
    vocab_inv = {v: k for k, v in tokenizer.vocab.items()}
    return [vocab_inv.get(tid, f"UNK_{tid}") for tid in token_ids]


def tokens_to_notes(token_ids):
    """
    Parse REMI token IDs into a structured note list.

    REMI structure: Bar [Position Pitch Velocity Duration]+ Bar ...
    Notes at the same Position share one Position token.

    Args:
        token_ids: list of int token IDs

    Returns:
        list of dicts, each with:
            'bar': int (0-indexed bar number)
            'position': int (Position token value, 0-31)
            'pitch': int (MIDI pitch, 21-109)
            'velocity_bin': int (velocity bin index, 0-31)
            'duration_idx': int (duration index, 0-63)
            'pitch_token': int (Pitch token ID)
            'velocity_token': int (Velocity token ID)
            'duration_token': int (Duration token ID)
            'position_token': int (Position token ID)
    """
    notes = []
    current_bar = -1
    current_position = None
    current_position_token = None

    i = 0
    while i < len(token_ids):
        tok = token_ids[i]

        if tok == BAR_TOKEN:
            current_bar += 1
            current_position = None
            current_position_token = None
            i += 1
        elif is_position_token(tok):
            current_position = tok - POSITION_OFFSET
            current_position_token = tok
            i += 1
        elif is_pitch_token(tok):
            # Expect Pitch, Velocity, Duration in sequence
            pitch_tok = tok
            midi_pitch = tok - PITCH_OFFSET + MIDI_PITCH_MIN

            vel_tok = token_ids[i + 1] if i + 1 < len(token_ids) else None
            dur_tok = token_ids[i + 2] if i + 2 < len(token_ids) else None

            if vel_tok is not None and is_velocity_token(vel_tok) and \
               dur_tok is not None and is_duration_token(dur_tok):
                notes.append({
                    'bar': current_bar,
                    'position': current_position if current_position is not None else 0,
                    'pitch': midi_pitch,
                    'velocity_bin': vel_tok - VELOCITY_OFFSET,
                    'duration_idx': dur_tok - DURATION_OFFSET,
                    'pitch_token': pitch_tok,
                    'velocity_token': vel_tok,
                    'duration_token': dur_tok,
                    'position_token': current_position_token if current_position_token is not None else POSITION_OFFSET,
                })
                i += 3
            else:
                i += 1
        else:
            i += 1

    return notes


def notes_to_tokens(notes, num_bars=None):
    """
    Convert structured note list back to REMI token IDs.

    Args:
        notes: list of note dicts (from tokens_to_notes)
        num_bars: total number of bars (if None, inferred from notes)

    Returns:
        list of int token IDs
    """
    if not notes:
        return [BAR_TOKEN]

    # Group notes by bar and position
    if num_bars is None:
        num_bars = max(n['bar'] for n in notes) + 1

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
        tokens.append(BAR_TOKEN)
        if bar_idx not in bars:
            continue
        # Sort positions
        for pos in sorted(bars[bar_idx].keys()):
            pos_token = POSITION_OFFSET + pos
            tokens.append(pos_token)
            # Sort notes by pitch within position
            pos_notes = sorted(bars[bar_idx][pos], key=lambda n: n['pitch'])
            for note in pos_notes:
                tokens.append(note['pitch_token'])
                tokens.append(note['velocity_token'])
                tokens.append(note['duration_token'])

    return tokens


def get_vocab_size():
    """Get REMI vocabulary size."""
    tokenizer = get_tokenizer()
    return len(tokenizer)


def get_duration_values():
    """Get list of duration event names for reference."""
    tokenizer = get_tokenizer()
    vocab = tokenizer.vocab
    durations = []
    for name, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if name.startswith('Duration_'):
            durations.append((idx, name))
    return durations


def get_velocity_values():
    """Get list of velocity event names for reference."""
    tokenizer = get_tokenizer()
    vocab = tokenizer.vocab
    velocities = []
    for name, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if name.startswith('Velocity_'):
            velocities.append((idx, name))
    return velocities
