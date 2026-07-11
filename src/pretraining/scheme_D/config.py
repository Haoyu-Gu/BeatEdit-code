"""
Music BERT configuration (absolute_bundled encoding scheme - Scheme D).

Based on absolute-position bundled encoding (absolute bundled encoding, vocab=7144).
Adds a [MASK] token on top of the original music vocabulary for MLM pretraining.

bundled_token = absolute_pitch x 81 + patch_value, range 0-7127.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """Music token encoding configuration (absolute_bundled scheme)."""
    # Patch encoding parameters
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 ternary patterns
    beats_length: int = 88    # 88 piano keys

    # Original vocabulary: 0-7143 (7144 tokens)
    original_vocab_size: int = 7144

    # bundled token range
    bundled_token_min: int = 0
    bundled_token_max: int = 7127

    # Special token IDs
    empty_marker_id: int = 7128    # empty-beat marker
    split_0_id: int = 7129         # upper-voice marker
    split_1_id: int = 7130         # lower-voice marker
    bar_token_id: int = 7131       # bar line
    eos_token_id: int = 7132
    bos_token_id: int = 7133
    pad_token_id: int = 7134
    time_sig_offset_id: int = 7135  # 7135-7139: 5 time signatures
    bpm_offset_id: int = 7140       # 7140-7143: 4 tempo buckets

    # Added for MLM
    mask_token_id: int = 7144      # [MASK] token
    vocab_size: int = 7145         # 7144 original + [MASK]

    # Note-token range (used to decide MLM masking)
    # Only bundled tokens (0-7127) are masked; control tokens (7128+) are not
    note_token_min: int = 0
    note_token_max: int = 7127

    @property
    def special_token_ids(self):
        """All special tokens that must not be masked."""
        ids = [
            self.empty_marker_id, self.split_0_id, self.split_1_id,
            self.bar_token_id, self.eos_token_id, self.bos_token_id,
            self.pad_token_id, self.mask_token_id,
        ]
        # Time-signature tokens
        ids.extend(range(self.time_sig_offset_id, self.time_sig_offset_id + 5))
        # BPM tokens
        ids.extend(range(self.bpm_offset_id, self.bpm_offset_id + 4))
        return set(ids)


@dataclass
class BertPretrainConfig:
    """BERT pretraining hyperparameters."""
    # Model architecture
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # MLM parameters
    mask_prob: float = 0.15       # 15% of note tokens are selected
    mask_replace_prob: float = 0.8   # 80% of those replaced with [MASK]
    mask_random_prob: float = 0.1    # 10% replaced with a random token
    # remaining 10% kept unchanged

    # Training parameters
    data_dir: str = "/path/to/data/npz"
    output_dir: str = "./checkpoints/music_bert_absolute_bundled"
    num_epochs: int = 30
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 2048
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # Logging and saving
    log_every_n_steps: int = 50
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    test_split_ratio: float = 0.05
    random_seed: int = 42

    def __post_init__(self):
        # Quick overrides for experimentation without editing this file, e.g.:
        #   BEATEDIT_LAYERS=4 BEATEDIT_HIDDEN=256 bash scripts/02_pretrain_bert.sh
        import os
        self.hidden_size = int(os.environ.get("BEATEDIT_HIDDEN", self.hidden_size))
        self.num_hidden_layers = int(os.environ.get("BEATEDIT_LAYERS", self.num_hidden_layers))
        self.num_attention_heads = int(os.environ.get("BEATEDIT_HEADS", self.num_attention_heads))
        self.intermediate_size = int(os.environ.get("BEATEDIT_FFN", self.intermediate_size))
        self.data_dir = os.environ.get("BEATEDIT_DATA_DIR", self.data_dir)
        self.num_epochs = int(os.environ.get("BEATEDIT_EPOCHS", self.num_epochs))
        self.batch_size = int(os.environ.get("BEATEDIT_BATCH", self.batch_size))


    # DataLoader
    num_workers: int = 4
