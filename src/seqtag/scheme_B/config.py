"""
Music GECToR Configuration

Token constants, label space definitions, and hyperparameters
for the GECToR-style music sequence correction system.

Encoding scheme: no_pair_related (relative position, vocab=185)
Label space: 350 labels (KEEP, DELETE, REPLACE, APPEND, SHIFT)
"""

# ==================== Token Constants (no_pair_related) ====================

# Patch value tokens: ternary encoding 3^4 = 81 patterns
PATCH_VALUE_MIN = 0
PATCH_VALUE_MAX = 80
NUM_PATCH_VALUES = 81

# Relative position tokens: 81 + relative_distance
POSITION_OFFSET = 81
POSITION_MIN = 81
POSITION_MAX = 168  # 81 + 87 (max piano key distance)

# Special tokens
EMPTY_MARKER = 169   # empty beat marker
END_MARKER = 170     # non-empty beat end marker
BAR_TOKEN = 171      # bar separator
EOS_TOKEN = 172      # end of sequence
BOS_TOKEN = 173      # beginning of sequence
PAD_TOKEN = 174      # padding

# Time signature tokens: 175-179 (5 types)
TIME_SIG_OFFSET = 175
NUM_TIME_SIGS = 5

# BPM tokens: 180-183 (4 categories)
BPM_OFFSET = 180
NUM_BPMS = 4

# MASK token (MLM pretraining only, not used in GECToR)
MASK_TOKEN = 184

# Vocabulary
VOCAB_SIZE = 185  # 0-184

# Token range boundaries
MUSIC_TOKEN_MIN = 0
MUSIC_TOKEN_MAX = 168
CONTROL_TOKEN_MIN = 169

# Maximum piano pitch index
MAX_PITCH = 87  # 0-87 = 88 keys


# ==================== Token Type Helpers ====================

def is_position_token(token):
    """Check if token is a relative position marker (81-168)."""
    return POSITION_MIN <= token <= POSITION_MAX


def is_patch_value(token):
    """Check if token is a patch value (0-80)."""
    return PATCH_VALUE_MIN <= token <= PATCH_VALUE_MAX


def is_control_token(token):
    """Check if token is a control/special token (>=169)."""
    return token >= CONTROL_TOKEN_MIN


def is_music_token(token):
    """Check if token represents actual music content (0-168)."""
    return MUSIC_TOKEN_MIN <= token <= MUSIC_TOKEN_MAX


def is_header_token(token):
    """Check if token belongs in the sequence header (BOS, TIME_SIG, BPM)."""
    return (token == BOS_TOKEN or
            TIME_SIG_OFFSET <= token < TIME_SIG_OFFSET + NUM_TIME_SIGS or
            BPM_OFFSET <= token < BPM_OFFSET + NUM_BPMS)


# ==================== Label Space (350 labels) ====================

NUM_LABELS = 350

# Label type ranges
LABEL_KEEP = 0
LABEL_DELETE = 1
LABEL_REPLACE_OFFSET = 2       # 2-170: REPLACE_0 .. REPLACE_168
LABEL_REPLACE_END = 170
LABEL_APPEND_OFFSET = 171      # 171-339: APPEND_0 .. APPEND_168
LABEL_APPEND_END = 339
LABEL_SHIFT_POS_OFFSET = 340   # 340-344: SHIFT +1 .. +5
LABEL_SHIFT_POS_END = 344
LABEL_SHIFT_NEG_OFFSET = 345   # 345-349: SHIFT -1 .. -5
LABEL_SHIFT_NEG_END = 349

LABEL_PAD = -100  # for padding positions in loss computation


# ==================== Label Helper Functions ====================

def label_id_keep():
    return LABEL_KEEP


def label_id_delete():
    return LABEL_DELETE


def label_id_replace(token_value):
    """REPLACE current token with music token `token_value` (0-168)."""
    assert 0 <= token_value <= MUSIC_TOKEN_MAX, \
        f"token_value must be in [0, {MUSIC_TOKEN_MAX}], got {token_value}"
    return LABEL_REPLACE_OFFSET + token_value


def label_id_append(token_value):
    """APPEND music token `token_value` (0-168) after current token."""
    assert 0 <= token_value <= MUSIC_TOKEN_MAX, \
        f"token_value must be in [0, {MUSIC_TOKEN_MAX}], got {token_value}"
    return LABEL_APPEND_OFFSET + token_value


def label_id_shift(delta):
    """SHIFT position token by `delta` (±1 to ±5)."""
    assert -5 <= delta <= 5 and delta != 0, \
        f"delta must be in [-5,-1]∪[1,5], got {delta}"
    if delta > 0:
        return 339 + delta   # 340-344
    else:
        return 344 + abs(delta)  # 345-349


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
        return ('SHIFT', label_id - 339)          # +1 to +5
    elif LABEL_SHIFT_NEG_OFFSET <= label_id <= LABEL_SHIFT_NEG_END:
        return ('SHIFT', -(label_id - 344))       # -1 to -5
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
DATA_DIR = "/path/to/data/npz"
