"""
Levenshtein Transformer for Music Inpainting - Configuration.

Token constants from with_pair encoding (Scheme C),
plus LevT-specific model and training hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

# ==================== Token Constants (with_pair, Scheme C) ====================

# Bundled token: relative_position * 81 + patch_value
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

# MASK token (used in MLM / inserter)
MASK_TOKEN = 7144

# Placeholder token for LevT (insert placeholder)
PLH_TOKEN = 7145  # [PLH] placeholder

# Base vocab + MASK + PLH
VOCAB_SIZE = 7146  # 0-7145

# Token range boundaries
CONTROL_TOKEN_MIN = 7128  # EMPTY_MARKER and above
NOTE_TOKEN_MAX = 7127     # max bundled note token

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


def is_note_token(token):
    """Check if token is a music content token (bundled note or EMPTY_MARKER or SPLIT)."""
    return is_bundled_token(token) or token == EMPTY_MARKER or is_split_token(token)


# ==================== LevT Model Configuration ====================

@dataclass
class LevTModelConfig:
    """Levenshtein Transformer model hyperparameters."""
    vocab_size: int = VOCAB_SIZE
    hidden_size: int = int(os.environ.get("BEATEDIT_HIDDEN", 512))
    num_hidden_layers: int = int(os.environ.get("BEATEDIT_LAYERS", 8))
    num_attention_heads: int = int(os.environ.get("BEATEDIT_HEADS", 8))
    intermediate_size: int = int(os.environ.get("BEATEDIT_FFN", 2048))
    max_position_embeddings: int = 2048
    dropout: float = 0.1

    # LevT-specific
    max_insert: int = 20      # max number of placeholders to insert per gap
    share_backbone: bool = True  # share encoder across all 3 heads

    # Special token IDs
    pad_token_id: int = PAD_TOKEN
    bos_token_id: int = BOS_TOKEN
    eos_token_id: int = EOS_TOKEN
    plh_token_id: int = PLH_TOKEN


# ==================== Training Configuration ====================

@dataclass
class LevTTrainingConfig:
    """Training hyperparameters."""
    # Data
    data_dir: str = "/path/to/data/npz"
    max_seq_len: int = 2048

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_ratio: float = 0.10
    gradient_accumulation_steps: int = 2
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0

    # Loss weights
    w_del: float = 1.0        # deletion loss weight
    w_ins: float = 1.0        # placeholder insertion loss weight
    w_tok: float = 1.0        # token prediction loss weight
    label_smoothing: float = 0.1  # for token prediction

    # Masking
    mask_beat_ratio_min: float = 0.125   # min fraction of beats to mask
    mask_beat_ratio_max: float = 0.50    # max fraction of beats to mask

    # Intermediate state corruption
    corruption_delete_prob: float = 0.3   # prob of deleting a token in mask region
    corruption_replace_prob: float = 0.2  # prob of replacing with random token
    # remaining 0.5 = keep correct

    # Inference
    max_iter: int = 10        # max iterative decoding steps

    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1     # save every N epochs

    # Paths
    output_dir: str = "checkpoints"
    checkpoint: str = None


# ==================== Data Path ====================

DATA_DIR = "/path/to/data/npz"
