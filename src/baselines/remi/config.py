"""
REMI GECToR Configuration

Token constants, label space definitions, and hyperparameters
for the GECToR-style music sequence correction system using REMI encoding.

REMI encoding: each note = Position + Pitch + Velocity + Duration (3-4 tokens)
Vocab size: 284 (miditok REMI with 32 velocity bins, no chords/tempo/time_sig)
Label space: 456 labels (KEEP, DELETE, REPLACE, APPEND, SHIFT)
"""

# ==================== Token Constants (REMI via miditok) ====================

# Special tokens
PAD_TOKEN = 0       # PAD_None
BOS_TOKEN = 1       # BOS_None
EOS_TOKEN = 2       # EOS_None
MASK_TOKEN = 3      # MASK_None
BAR_TOKEN = 4       # Bar_None

# Pitch tokens: 5-93 (MIDI pitch 21-109, 89 keys)
PITCH_OFFSET = 5
PITCH_MIN = 5
PITCH_MAX = 93
NUM_PITCHES = 89
MIDI_PITCH_MIN = 21
MIDI_PITCH_MAX = 109

# Velocity tokens: 94-125 (32 velocity bins)
VELOCITY_OFFSET = 94
VELOCITY_MIN = 94
VELOCITY_MAX = 125
NUM_VELOCITIES = 32

# Duration tokens: 126-189 (64 duration values)
DURATION_OFFSET = 126
DURATION_MIN = 126
DURATION_MAX = 189
NUM_DURATIONS = 64

# Position tokens: 190-221 (32 positions within bar, 0-31)
POSITION_OFFSET = 190
POSITION_MIN = 190
POSITION_MAX = 221
NUM_POSITIONS = 32

# PitchDrum tokens: 222-283 (not used for piano, but in vocab)
PITCH_DRUM_OFFSET = 222
PITCH_DRUM_MIN = 222
PITCH_DRUM_MAX = 283

# Vocabulary
VOCAB_SIZE = 284

# Music token range (tokens that represent actual musical content)
MUSIC_TOKEN_MIN = 5      # Pitch_21
MUSIC_TOKEN_MAX = 221    # Position_31

# ==================== Token Type Helpers ====================


def is_pitch_token(token):
    """Check if token is a Pitch token (5-93)."""
    return PITCH_MIN <= token <= PITCH_MAX


def is_velocity_token(token):
    """Check if token is a Velocity token (94-125)."""
    return VELOCITY_MIN <= token <= VELOCITY_MAX


def is_duration_token(token):
    """Check if token is a Duration token (126-189)."""
    return DURATION_MIN <= token <= DURATION_MAX


def is_position_token(token):
    """Check if token is a Position token (190-221)."""
    return POSITION_MIN <= token <= POSITION_MAX


def is_music_token(token):
    """Check if token represents actual music content (5-221)."""
    return MUSIC_TOKEN_MIN <= token <= MUSIC_TOKEN_MAX


def is_note_content_token(token):
    """Check if token is Pitch, Velocity, or Duration (5-189, not Position)."""
    return PITCH_MIN <= token <= DURATION_MAX


def is_special_token(token):
    """Check if token is a special/structural token (PAD, BOS, EOS, MASK, Bar)."""
    return token <= BAR_TOKEN


def is_maskable_token(token):
    """Check if token should be masked during MLM pretraining (music tokens only)."""
    return is_music_token(token)


# ==================== Label Space (456 labels) ====================
# KEEP(1) + DELETE(1) + REPLACE_0-221(222) + APPEND_0-221(222) + SHIFT_±1-5(10)

NUM_LABELS = 456

# Label type ranges
LABEL_KEEP = 0
LABEL_DELETE = 1
LABEL_REPLACE_OFFSET = 2           # 2-223: REPLACE_0 .. REPLACE_221
LABEL_REPLACE_END = 223
LABEL_APPEND_OFFSET = 224          # 224-445: APPEND_0 .. APPEND_221
LABEL_APPEND_END = 445
LABEL_SHIFT_POS_OFFSET = 446       # 446-450: SHIFT +1 .. +5
LABEL_SHIFT_POS_END = 450
LABEL_SHIFT_NEG_OFFSET = 451       # 451-455: SHIFT -1 .. -5
LABEL_SHIFT_NEG_END = 455

LABEL_PAD = -100  # for padding positions in loss computation


# ==================== Label Helper Functions ====================

def label_id_keep():
    return LABEL_KEEP


def label_id_delete():
    return LABEL_DELETE


def label_id_replace(token_value):
    """REPLACE current token with token `token_value` (0-221)."""
    assert 0 <= token_value <= MUSIC_TOKEN_MAX, \
        f"token_value must be in [0, {MUSIC_TOKEN_MAX}], got {token_value}"
    return LABEL_REPLACE_OFFSET + token_value


def label_id_append(token_value):
    """APPEND token `token_value` (0-221) after current token."""
    assert 0 <= token_value <= MUSIC_TOKEN_MAX, \
        f"token_value must be in [0, {MUSIC_TOKEN_MAX}], got {token_value}"
    return LABEL_APPEND_OFFSET + token_value


def label_id_shift(delta):
    """SHIFT position token by `delta` (+/-1 to +/-5)."""
    assert -5 <= delta <= 5 and delta != 0, \
        f"delta must be in [-5,-1] U [1,5], got {delta}"
    if delta > 0:
        return LABEL_SHIFT_POS_OFFSET + delta - 1   # 446-450
    else:
        return LABEL_SHIFT_NEG_OFFSET + abs(delta) - 1  # 451-455


def decode_label(label_id):
    """Decode label ID to (operation, value) tuple."""
    if label_id == LABEL_KEEP:
        return ('KEEP', None)
    elif label_id == LABEL_DELETE:
        return ('DELETE', None)
    elif LABEL_REPLACE_OFFSET <= label_id <= LABEL_REPLACE_END:
        return ('REPLACE', label_id - LABEL_REPLACE_OFFSET)
    elif LABEL_APPEND_OFFSET <= label_id <= LABEL_APPEND_END:
        return ('APPEND', label_id - LABEL_APPEND_OFFSET)
    elif LABEL_SHIFT_POS_OFFSET <= label_id <= LABEL_SHIFT_POS_END:
        return ('SHIFT', label_id - LABEL_SHIFT_POS_OFFSET + 1)    # +1 to +5
    elif LABEL_SHIFT_NEG_OFFSET <= label_id <= LABEL_SHIFT_NEG_END:
        return ('SHIFT', -(label_id - LABEL_SHIFT_NEG_OFFSET + 1))  # -1 to -5
    else:
        raise ValueError(f"Invalid label ID: {label_id}")


# ==================== Perturbation Defaults ====================

DEFAULT_PERTURB_PROBS = {
    'p_pitch': 0.10,
    'p_rhythm': 0.05,
    'p_delete': 0.03,
    'p_insert': 0.02,
}

DIFFICULTY_PRESETS = {
    'easy':   {'p_pitch': 0.15, 'p_rhythm': 0.08, 'p_delete': 0.05, 'p_insert': 0.03},
    'medium': {'p_pitch': 0.10, 'p_rhythm': 0.05, 'p_delete': 0.03, 'p_insert': 0.02},
    'hard':   {'p_pitch': 0.05, 'p_rhythm': 0.02, 'p_delete': 0.01, 'p_insert': 0.01},
}


# ==================== Training Hyperparameters ====================

TRAINING_DEFAULTS = {
    # Model
    'num_labels': NUM_LABELS,
    'dropout': 0.1,

    # Stage I
    'stage1_epochs': 20,
    'freeze_epochs': 2,
    'cold_lr': 1e-3,
    'finetune_lr_bert': 1e-5,
    'finetune_lr_head': 1e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.10,
    'batch_size_per_gpu': 32,
    'max_seq_len': 2048,
    'keep_weight': 0.15,
    'lambda_detect': 0.5,
    'early_stopping_patience': 3,

    # Stage III
    'stage3_epochs': 3,
    'stage3_lr': 5e-6,
    'clean_ratio': 0.25,

    # Inference
    'max_iterations': 3,
    'keep_confidence_bias': 0.3,
    'error_threshold': 0.5,
}

# Data
MIDI_DATA_DIR = "/path/to/data/midi"
NPZ_DATA_DIR = "/path/to/data/npz"
