"""
Music BERT configuration.

Based on the no_pair_related encoding scheme (relative position encoding, vocab=184).
Adds a [MASK] token on top of the original music vocabulary for MLM pretraining.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """Music token encoding configuration (no_pair_related scheme)."""
    # Patch encoding parameters
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 ternary patterns
    beats_length: int = 88    # 88 piano keys

    # Original vocabulary: 0-183 (184 tokens)
    # patch value tokens: 0-80
    # relative position tokens: 81-168 (81 + relative_distance)
    # special tokens: 169-183
    original_vocab_size: int = 184

    # Special token IDs
    empty_marker_id: int = 169    # empty-beat marker
    end_marker_id: int = 170      # end-of-non-empty-beat marker
    bar_token_id: int = 171       # bar line
    eos_token_id: int = 172
    bos_token_id: int = 173
    pad_token_id: int = 174
    time_sig_offset_id: int = 175  # 175-179: 5 time signatures
    bpm_offset_id: int = 180       # 180-183: 4 tempo buckets

    # Added for MLM
    mask_token_id: int = 184      # [MASK] token
    vocab_size: int = 185         # original 184 + [MASK]

    # Note-token range (used to decide MLM masking)
    # Covers patch values (0-80) and relative positions (81-168)
    note_token_min: int = 0
    note_token_max: int = 168

    @property
    def special_token_ids(self):
        """All special tokens that must not be masked."""
        ids = [
            self.empty_marker_id, self.end_marker_id,
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
    output_dir: str = "./checkpoints/music_bert"
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
        self.mixed_precision = os.environ.get("BEATEDIT_PRECISION", self.mixed_precision)


    # DataLoader
    num_workers: int = 4
