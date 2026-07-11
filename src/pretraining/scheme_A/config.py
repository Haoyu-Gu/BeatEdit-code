"""
Music BERT configuration.

Based on the no_pair encoding scheme (absolute position encoding, vocab=185).
Adds a [MASK] token on top of the original music vocabulary for MLM pretraining.

Note: the original LLaMA model had vocab_size=268 (with reserved slots); the
    actual maximum token ID is 184. BERT uses a compact vocabulary:
    0-184 + [MASK]=185, for 186 total.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """Music token encoding configuration (no_pair scheme - absolute position)."""
    # Patch encoding parameters
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 ternary patterns
    beats_length: int = 88    # 88 piano keys

    # Original vocabulary: 0-184 actually used (185 tokens)
    # patch value tokens: 0-80
    # absolute position tokens: 81-168 (81 + pitch_index)
    # (token 169 unused)
    # special tokens: 170-184
    original_vocab_size: int = 185

    # Special token IDs (the no_pair scheme has no EMPTY/END marker)
    bar_token_id: int = 170
    eos_token_id: int = 171
    bos_token_id: int = 172
    pad_token_id: int = 173
    time_sig_offset_id: int = 174  # 174-178: 5 time signatures
    bpm_offset_id: int = 179       # 179-182: 4 tempo buckets
    track0_start_id: int = 183     # track 0 start marker
    track1_start_id: int = 184     # track 1 start marker

    # Added for MLM
    mask_token_id: int = 185      # [MASK] token
    vocab_size: int = 186         # 185 real tokens + [MASK]

    # Note-token range (used to decide MLM masking)
    # Covers patch values (0-80) and absolute positions (81-168)
    note_token_min: int = 0
    note_token_max: int = 168

    @property
    def special_token_ids(self):
        """All special tokens that must not be masked."""
        ids = [
            self.bar_token_id, self.eos_token_id, self.bos_token_id,
            self.pad_token_id, self.mask_token_id,
            self.track0_start_id, self.track1_start_id,
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
    output_dir: str = "./checkpoints/music_bert_no_pair"
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
