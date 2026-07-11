"""
FELIX-Music Configuration (Scheme B: no_pair_related)

Token constants (from no_pair_related encoding), FELIX token-level label system,
and model/training hyperparameters.

Encoding scheme: no_pair_related (relative position, 2 tokens per note, vocab=185)
  Each note = [position_token][value_token], beats end with END_MARKER
  Position tokens: 81-168 (81 + relative_distance)
  Value tokens: 0-80 (81 ternary patterns)
FELIX label space: 11 token-level labels (KEEP/DELETE/REPLACE/APPEND_1..8)
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

# ==================== Token Constants (no_pair_related) ====================

# Patch value tokens: ternary encoding 3^4 = 81 patterns
PATCH_VALUE_MIN = 0
PATCH_VALUE_MAX = 80
NUM_PATCH_VALUES = 81

# Pattern number (3^4 = 81)
PATTERN_NUM = 81

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

# MASK token
MASK_TOKEN = 184

# Vocabulary
VOCAB_SIZE = 185  # 0-184

# Token range boundaries
MUSIC_TOKEN_MIN = 0
MUSIC_TOKEN_MAX = 168
CONTROL_TOKEN_MIN = 169  # EMPTY_MARKER and above

# Maximum piano pitch index
MAX_PITCH = 87  # 0-87 = 88 keys


# ==================== Token Type Helpers ====================

def is_position_token(token):
    """Check if token is a relative position marker (81-168)."""
    return POSITION_MIN <= token <= POSITION_MAX


def is_patch_value(token):
    """Check if token is a patch value (0-80)."""
    return PATCH_VALUE_MIN <= token <= PATCH_VALUE_MAX


def is_music_token(token):
    """Check if token represents actual music content (0-168)."""
    return MUSIC_TOKEN_MIN <= token <= MUSIC_TOKEN_MAX


def is_control_token(token):
    """Check if token is a control/special token (>=169)."""
    return token >= CONTROL_TOKEN_MIN


def is_header_token(token):
    """Check if token belongs in the sequence header (BOS, TIME_SIG, BPM)."""
    return (token == BOS_TOKEN or
            TIME_SIG_OFFSET <= token < TIME_SIG_OFFSET + NUM_TIME_SIGS or
            BPM_OFFSET <= token < BPM_OFFSET + NUM_BPMS)


# ==================== FELIX Token-Level Label Space (11 labels) ====================
#
# Like GECToR: one label per token. Tagger only decides structure,
# Inserter fills actual content at MASK positions.
#
# KEEP=0          — keep this token unchanged
# DELETE=1        — delete this token
# REPLACE=2       — replace this token with 1 MASK (Inserter fills it)
# APPEND_1..8     — keep this token, insert 1..8 MASKs after (3..10)
# NUM_FELIX_LABELS=11

LABEL_KEEP = 0
LABEL_DELETE = 1
LABEL_REPLACE = 2

LABEL_APPEND_OFFSET = 3  # APPEND_1=3, APPEND_2=4, ..., APPEND_8=10
LABEL_APPEND_MAX_N = 8

NUM_FELIX_LABELS = 11

LABEL_PAD = -100  # for padding positions in loss computation


# ==================== FELIX Label Helper Functions ====================

def label_id_keep():
    return LABEL_KEEP


def label_id_delete():
    return LABEL_DELETE


def label_id_replace():
    return LABEL_REPLACE


def label_id_append(n):
    """APPEND label: keep this token + insert n MASKs after (n=1..8)."""
    assert 1 <= n <= LABEL_APPEND_MAX_N, f"n must be in [1, {LABEL_APPEND_MAX_N}], got {n}"
    return LABEL_APPEND_OFFSET + n - 1


def decode_felix_label(label_id):
    """Decode FELIX label ID to (operation, value) tuple."""
    if label_id == LABEL_KEEP:
        return ('KEEP', None)
    elif label_id == LABEL_DELETE:
        return ('DELETE', None)
    elif label_id == LABEL_REPLACE:
        return ('REPLACE', None)
    elif LABEL_APPEND_OFFSET <= label_id <= LABEL_APPEND_OFFSET + LABEL_APPEND_MAX_N - 1:
        n = label_id - LABEL_APPEND_OFFSET + 1
        return ('APPEND', n)
    else:
        raise ValueError(f"Invalid FELIX label ID: {label_id}")


# ==================== Model Configurations ====================

@dataclass
class TaggerConfig:
    """Tagger model hyperparameters (Transformer Encoder + per-token classification)."""
    vocab_size: int = VOCAB_SIZE
    hidden_size: int = int(os.environ.get("BEATEDIT_HIDDEN", 512))
    num_hidden_layers: int = int(os.environ.get("BEATEDIT_LAYERS", 8))
    num_attention_heads: int = int(os.environ.get("BEATEDIT_HEADS", 8))
    intermediate_size: int = int(os.environ.get("BEATEDIT_FFN", 2048))
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    num_labels: int = NUM_FELIX_LABELS


@dataclass
class InserterConfig:
    """Inserter model hyperparameters (MLM-style)."""
    vocab_size: int = VOCAB_SIZE
    hidden_size: int = int(os.environ.get("BEATEDIT_HIDDEN", 512))
    num_hidden_layers: int = int(os.environ.get("BEATEDIT_LAYERS", 8))
    num_attention_heads: int = int(os.environ.get("BEATEDIT_HEADS", 8))
    intermediate_size: int = int(os.environ.get("BEATEDIT_FFN", 2048))
    max_position_embeddings: int = 2048
    dropout: float = 0.1


@dataclass
class FELIXTrainingConfig:
    """Training hyperparameters for FELIX pipeline."""
    # Data
    data_dir: str = "/path/to/data/npz"
    max_seq_len: int = 2048

    # Tagger training
    tagger_batch_size: int = 32
    tagger_lr: float = 1e-4
    tagger_epochs: int = 30
    tagger_warmup_ratio: float = 0.10
    tagger_weight_decay: float = 0.01

    # Inserter training
    inserter_batch_size: int = 32
    inserter_lr: float = 1e-4
    inserter_epochs: int = 30
    inserter_warmup_ratio: float = 0.10
    inserter_weight_decay: float = 0.01

    # Perturbation level weights (L1, L2, L3, L4)
    level_weights: Tuple[int, ...] = (30, 30, 25, 15)

    # Gradient accumulation
    gradient_accumulation_steps: int = 3

    # Mixed precision
    mixed_precision: str = "fp16"


# ==================== Data Path ====================

DATA_DIR = "/path/to/data/npz"
