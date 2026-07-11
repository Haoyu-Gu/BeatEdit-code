"""
CP (Compound Word) GECToR Configuration

Sub-vocabulary constants, action label definitions, and hyperparameters
for the GECToR-style music sequence correction system using CPWord encoding.

CPWord encoding: each compound token = [family, position, pitch, velocity, duration]
- Bar token: [Metric, Bar, Ignore, Ignore, Ignore]
- Position token: [Metric, Position_X, Ignore, Ignore, Ignore]
- Note token: [Note, Ignore, Pitch_P, Vel_V, Dur_D]
"""

# ==================== Sub-vocabulary Sizes (from miditok CPWord) ====================

NUM_SUBVOCABS = 5

# Sub-vocab 0: Family
FAMILY_VOCAB_SIZE = 6    # PAD(0), BOS(1), EOS(2), MASK(3), Metric(4), Note(5)
FAMILY_PAD = 0
FAMILY_BOS = 1
FAMILY_EOS = 2
FAMILY_MASK = 3
FAMILY_METRIC = 4
FAMILY_NOTE = 5

# Sub-vocab 1: Position
POSITION_VOCAB_SIZE = 38  # PAD(0), BOS(1), EOS(2), MASK(3), Ignore(4), Bar(5), Position_0..31(6-37)
POS_PAD = 0
POS_BOS = 1
POS_EOS = 2
POS_MASK = 3
POS_IGNORE = 4
POS_BAR = 5
POS_OFFSET = 6      # Position_0 starts at 6
POS_MAX = 37         # Position_31
NUM_POSITIONS = 32   # 0..31

# Sub-vocab 2: Pitch
PITCH_VOCAB_SIZE = 156  # PAD(0), BOS(1), EOS(2), MASK(3), Ignore(4), Pitch_21..109(5-93), PitchDrum(94-155)
PITCH_PAD = 0
PITCH_BOS = 1
PITCH_EOS = 2
PITCH_MASK = 3
PITCH_IGNORE = 4
PITCH_OFFSET = 5        # Pitch_21 starts at 5
PITCH_MIN_ID = 5        # Pitch_21
PITCH_MAX_ID = 93       # Pitch_109
NUM_PITCHES = 89         # 21..109
MIDI_PITCH_MIN = 21
MIDI_PITCH_MAX = 109

# Sub-vocab 3: Velocity
VELOCITY_VOCAB_SIZE = 37  # PAD(0), BOS(1), EOS(2), MASK(3), Ignore(4), Vel_3..127(5-36)
VEL_PAD = 0
VEL_BOS = 1
VEL_EOS = 2
VEL_MASK = 3
VEL_IGNORE = 4
VEL_OFFSET = 5          # first velocity bin
VEL_MAX_ID = 36
NUM_VELOCITIES = 32

# Sub-vocab 4: Duration
DURATION_VOCAB_SIZE = 69  # PAD(0), BOS(1), EOS(2), MASK(3), Ignore(4), Dur(5-68)
DUR_PAD = 0
DUR_BOS = 1
DUR_EOS = 2
DUR_MASK = 3
DUR_IGNORE = 4
DUR_OFFSET = 5
DUR_MAX_ID = 68
NUM_DURATIONS = 64

# All sub-vocab sizes as tuple (for model)
SUB_VOCAB_SIZES = (
    FAMILY_VOCAB_SIZE,    # 6
    POSITION_VOCAB_SIZE,  # 38
    PITCH_VOCAB_SIZE,     # 156
    VELOCITY_VOCAB_SIZE,  # 37
    DURATION_VOCAB_SIZE,  # 69
)

# Common IDs across all sub-vocabs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
IGNORE_ID = 4

# Special compound tokens
BOS_TOKEN = [FAMILY_BOS, POS_BOS, PITCH_BOS, VEL_BOS, DUR_BOS]
EOS_TOKEN = [FAMILY_EOS, POS_EOS, PITCH_EOS, VEL_EOS, DUR_EOS]
PAD_TOKEN = [FAMILY_PAD, POS_PAD, PITCH_PAD, VEL_PAD, DUR_PAD]
MASK_COMPOUND = [FAMILY_MASK, POS_MASK, PITCH_MASK, VEL_MASK, DUR_MASK]


# ==================== Token Type Helpers ====================

def is_bar_token(compound):
    """Check if compound token is a Bar token."""
    return compound[0] == FAMILY_METRIC and compound[1] == POS_BAR

def is_position_token(compound):
    """Check if compound token is a Position marker."""
    return compound[0] == FAMILY_METRIC and POS_OFFSET <= compound[1] <= POS_MAX

def is_note_token(compound):
    """Check if compound token is a Note."""
    return compound[0] == FAMILY_NOTE

def is_bos_token(compound):
    return compound == BOS_TOKEN

def is_eos_token(compound):
    return compound == EOS_TOKEN

def is_pad_token(compound):
    return compound[0] == PAD_ID

def is_special_token(compound):
    """Check if compound token is BOS, EOS, or PAD."""
    return compound[0] in (FAMILY_PAD, FAMILY_BOS, FAMILY_EOS)

def is_music_token(compound):
    """Check if compound token is a music content token (Bar, Position, or Note)."""
    return compound[0] in (FAMILY_METRIC, FAMILY_NOTE)

def get_position_value(compound):
    """Get the position value (0-31) from a Position compound token."""
    return compound[1] - POS_OFFSET

def get_midi_pitch(compound):
    """Get MIDI pitch (21-109) from a Note compound token."""
    return compound[2] - PITCH_OFFSET + MIDI_PITCH_MIN


# ==================== Action Labels (Factored) ====================

ACTION_KEEP = 0
ACTION_DELETE = 1
ACTION_REPLACE = 2
ACTION_APPEND = 3
NUM_ACTIONS = 4

LABEL_PAD = -100  # for padding positions in loss computation


# ==================== Perturbation Defaults ====================

DEFAULT_PERTURB_PROBS = {
    'p_pitch': 0.10,
    'p_rhythm': 0.05,
    'p_delete': 0.03,
    'p_insert': 0.02,
}


# ==================== Training Hyperparameters ====================

TRAINING_DEFAULTS = {
    # Model
    'hidden_size': 512,
    'num_hidden_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 2048,
    'max_position_embeddings': 2048,
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
    'lambda_sub': 1.0,
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
