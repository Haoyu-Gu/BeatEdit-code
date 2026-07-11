"""
Music GECToR Configuration (with_pair encoding)

Token constants, label space definitions, and hyperparameters
for the GECToR-style music sequence correction system.

Encoding scheme: with_pair bundled encoding (vocab=7145 with MASK)
Label space: 14258 labels (KEEP, DELETE, REPLACE_0..7127, APPEND_0..7127)
"""

# ==================== Token Constants (with_pair) ====================

# Bundled token: relative_position × 81 + patch_value
# Range: 0-7127
BUNDLED_TOKEN_MIN = 0
BUNDLED_TOKEN_MAX = 7127
NUM_BUNDLED_TOKENS = 7128

# Pattern number (3^4 = 81)
PATTERN_NUM = 81

# Special tokens
EMPTY_MARKER = 7128   # empty beat marker
SPLIT_0 = 7129        # high voice (Track 0) beat start
SPLIT_1 = 7130        # low voice (Track 1) beat start
BAR_TOKEN = 7131      # bar separator
EOS_TOKEN = 7132      # end of sequence
BOS_TOKEN = 7133      # beginning of sequence
PAD_TOKEN = 7134      # padding

# Time signature tokens: 7135-7139 (5 types)
TIME_SIG_OFFSET = 7135
NUM_TIME_SIGS = 5

# BPM tokens: 7140-7143 (4 categories)
BPM_OFFSET = 7140
NUM_BPMS = 4

# MASK token (MLM pretraining only)
MASK_TOKEN = 7144

# Vocabulary
VOCAB_SIZE = 7145  # 0-7144

# Token range boundaries
CONTROL_TOKEN_MIN = 7128  # EMPTY_MARKER and above

# Maximum piano pitch index
MAX_PITCH = 87  # 0-87 = 88 keys


# ==================== Token Type Helpers ====================

def is_bundled_token(token):
    """Check if token is a bundled note token (0-7127)."""
    return BUNDLED_TOKEN_MIN <= token <= BUNDLED_TOKEN_MAX


def is_control_token(token):
    """Check if token is a control/special token (>=7128)."""
    return token >= CONTROL_TOKEN_MIN


def is_split_token(token):
    """Check if token is a SPLIT marker (7129 or 7130)."""
    return token == SPLIT_0 or token == SPLIT_1


def is_header_token(token):
    """Check if token belongs in the sequence header (BOS, TIME_SIG, BPM)."""
    return (token == BOS_TOKEN or
            TIME_SIG_OFFSET <= token < TIME_SIG_OFFSET + NUM_TIME_SIGS or
            BPM_OFFSET <= token < BPM_OFFSET + NUM_BPMS)


# ==================== Label Space (14258 labels, Scheme A) ====================
# 0:         KEEP
# 1:         DELETE
# 2-7129:    REPLACE_0 .. REPLACE_7127  (replace with bundled token x)
# 7130-14257: APPEND_0 .. APPEND_7127   (append bundled token x after current)
# Total: 14258

NUM_LABELS = 14258

# Label type ranges
LABEL_KEEP = 0
LABEL_DELETE = 1
LABEL_REPLACE_OFFSET = 2          # REPLACE_x = 2 + x
LABEL_REPLACE_END = 7129          # 2 + 7127
LABEL_APPEND_OFFSET = 7130        # APPEND_x = 7130 + x
LABEL_APPEND_END = 14257          # 7130 + 7127

LABEL_PAD = -100  # for padding positions in loss computation


# ==================== Label Helper Functions ====================

def label_id_keep():
    return LABEL_KEEP


def label_id_delete():
    return LABEL_DELETE


def label_id_replace(bundled_token):
    """REPLACE current token with bundled_token (0-7127)."""
    assert 0 <= bundled_token <= BUNDLED_TOKEN_MAX, \
        f"bundled_token must be in [0, {BUNDLED_TOKEN_MAX}], got {bundled_token}"
    return LABEL_REPLACE_OFFSET + bundled_token


def label_id_append(bundled_token):
    """APPEND bundled_token (0-7127) after current token."""
    assert 0 <= bundled_token <= BUNDLED_TOKEN_MAX, \
        f"bundled_token must be in [0, {BUNDLED_TOKEN_MAX}], got {bundled_token}"
    return LABEL_APPEND_OFFSET + bundled_token


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
    'batch_size_per_gpu': 16,
    'max_seq_len': 2048,
    'keep_weight': 0.15,
    'lambda_detect': 0.5,
    'early_stopping_patience': 3,

    # Stage III
    'stage3_epochs': 3,
    'stage3_lr': 5e-6,
    'clean_ratio': 0.25,

    # Inference
    'max_iterations': 2,          # with_pair 需要更少迭代
    'keep_confidence_bias': 0.3,
    'error_threshold': 0.5,
}

# Data
DATA_DIR = "/path/to/data/npz"
