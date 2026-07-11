from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Base configuration
    time = datetime.now().strftime("%m%d_%H%M")
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    save_model_epochs: int = 1
    train_batch_size = 2
    use_length_aware_batching = True  # whether to use length-aware batching
    gradient_accumulation_steps: int = 32
    lr_warmup_steps: int = 50
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # Learning-rate configuration
    learning_rate: float = 5e-5

    # Logging configuration
    log: bool = True
    log_every_n_steps: int = 20
    tensorboard_log_dir: str = "/path/to/logs/encoding"
    tensorboard_log_name: str = "no_pair_related"
    data_dir = "/path/to/data/npz"
    save_steps = 60000

    # Test-set configuration
    use_test_set: bool = True  # whether to use a test set
    test_split_ratio: float = 0.10  # test-set split ratio
    test_frequency: float = 0.20  # test frequency (fraction of an epoch between tests)
    test_batch_size: int = 1  # batch size used during testing
    test_save_results: bool = True  # whether to save test results to tensorboard
    random_seed: int = 42  # random seed for the dataset split


@dataclass
class ModelConfig:
    """Model architecture configuration.

    NOTE (release): only the token-protocol fields of this class (patch_h,
    patch_w, pattern_num, beats_length, vocab/marker ids, train_cutoff_len)
    are used by the encoding pipeline (PianoDataset / token2midi).
    The transformer hyperparameters below (hidden_size=768, 16 layers, RoPE)
    belong to a legacy autoregressive generator and are NOT the BeatEdit
    backbone -- the paper's Music BERT config lives in
    src/pretraining/scheme_*/config.py (512 hidden / 8 layers / 8 heads).
    """
    # Model configuration
    hidden_size: int = 768  #
    num_hidden_layers: int = 16
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 3000
    train_cutoff_len = 2048  # sequence truncation length during training
    min_length = -1
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # Relative position encoding (no_pair_related)
    # 0-80: patch token values (ternary encoding)
    # 81-168: position markers (81 + relative_distance, max distance=87)
    # 169: empty_marker (empty beat)
    # 170: end_marker (end of a non-empty beat)
    # 171+: special tokens
    vocab_size: int = 184
    patch_h = 1  # patch size along the pitch axis
    patch_w = 4  # patch size along the time axis

    pattern_num = 81   # position-marker offset (position_marker = 81 + relative_pos)
    beats_length = 88  # number of piano keys (88 keys)

    # Special tokens
    empty_marker_id: int = 169    # empty-beat marker
    end_marker_id: int = 170      # end-of-non-empty-beat marker

    bar_token_id: int = 171
    eos_token_id: int = 172
    bos_token_id: int = 173
    pad_token_id: int = 174

    time_sig_offset_id: int = 175    # '4/4': +0, '3/4': +1, '2/4': +2, '6/8': +3, '2/2': +4  (175~179)
    bpm_offset_id: int = 180         # <90: +0, 90~200: +1, >200: +2, UNK: +3  (180~183)
