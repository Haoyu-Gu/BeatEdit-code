from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    time = datetime.now().strftime("%m%d_%H%M")
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    save_model_epochs: int = 1
    train_batch_size = 2
    use_length_aware_batching = True  # whether to use length-aware batching
    gradient_accumulation_steps: int = 32
    lr_warmup_steps: int = 50
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # Learning-rate settings
    learning_rate: float = 5e-5

    # Logging settings
    log: bool = True
    log_every_n_steps: int = 20
    tensorboard_log_dir: str = "/path/to/logs/encoding"
    tensorboard_log_name: str = "no_pair"
    data_dir = "/path/to/data/npz"
    save_steps = 60000

    # Test-set settings
    use_test_set: bool = True  # whether to use a test set
    test_split_ratio: float = 0.10  # fraction of data held out for testing
    test_frequency: float = 0.20  # test frequency (once every 0.20 epoch)
    test_batch_size: int = 1  # batch size used during testing
    test_save_results: bool = True  # whether to log test results to tensorboard
    random_seed: int = 42  # random seed for the train/test split


@dataclass
class ModelConfig:
    """Model-architecture configuration.

    NOTE (release): only the token-protocol fields of this class (patch_h,
    patch_w, pattern_num, beats_length, vocab/marker ids, train_cutoff_len)
    are used by the encoding pipeline (PianoDataset / token2midi).
    The transformer hyperparameters below (hidden_size=768, 16 layers, RoPE)
    belong to a legacy autoregressive generator and are NOT the BeatEdit
    backbone -- the paper's Music BERT config lives in
    src/pretraining/scheme_*/config.py (512 hidden / 8 layers / 8 heads).
    """
    # Model settings
    hidden_size: int = 768  #
    num_hidden_layers: int = 16
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 3000
    train_cutoff_len = 2048  # truncation length used during training
    min_length = -1
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # Encoding protocol markoffset | measures_length | bar_token_ids | time_sig_ids | bpm_ids | eos_token_id | bos_token_id | pad_token_id | track_start_ids
    vocab_size: int = 268
    patch_h = 1  # patch size along the pitch axis
    patch_w = 4  # patch size along the time axis

    pattern_num = 81  # start ID for position tokens
    beats_length = 88  # max length per bar (in number of position tokens) 169 -> 170

    bar_token_id: int = 170
    eos_token_id: int = 171
    bos_token_id: int = 172
    pad_token_id: int = 173

    time_sig_offset_id: int = 174    # '4/4':  +0 , '3/4': +1, '2/4':  +2, '6/8': +3, '2/2': +4  178 ->179
    bpm_offset_id: int = 179         #  <90  : +0 ,  <200 : +1   ,  >300 : +2  ,  unkown : +3    182 ->183

    # Reserved track-start ids for program (128 possible in theory; 2 used currently)
    track0_start_id = 183
    track1_start_id = 184
