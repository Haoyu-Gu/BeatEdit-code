"""
Music BERT 配置 (with_pair 编码方案)

基于 with_pair 捆绑编码 (bundled encoding, vocab=7144)
在原始音乐词表基础上增加 [MASK] token 用于 MLM 预训练

bundled_token = relative_position × 81 + patch_value, 范围 0-7127
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """音乐 token 编码配置 (with_pair 方案)"""
    # Patch 编码参数
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 种三进制 pattern
    beats_length: int = 88    # 88 钢琴键

    # 原始词表: 0-7143 (7144 tokens)
    original_vocab_size: int = 7144

    # bundled token 范围
    bundled_token_min: int = 0
    bundled_token_max: int = 7127

    # 特殊 token IDs
    empty_marker_id: int = 7128    # 空beat标记
    split_0_id: int = 7129         # 高声部标记
    split_1_id: int = 7130         # 低声部标记
    bar_token_id: int = 7131       # 小节线
    eos_token_id: int = 7132
    bos_token_id: int = 7133
    pad_token_id: int = 7134
    time_sig_offset_id: int = 7135  # 7135-7139: 5种拍号
    bpm_offset_id: int = 7140       # 7140-7143: 4种速度

    # MLM 新增
    mask_token_id: int = 7144      # [MASK] token
    vocab_size: int = 7145         # 原始 7144 + [MASK]

    # Note token 范围 (用于 MLM masking 判断)
    # 只 mask bundled token (0-7127)，不 mask 控制 token (7128+)
    note_token_min: int = 0
    note_token_max: int = 7127

    @property
    def special_token_ids(self):
        """所有不应被 mask 的特殊 token"""
        ids = [
            self.empty_marker_id, self.split_0_id, self.split_1_id,
            self.bar_token_id, self.eos_token_id, self.bos_token_id,
            self.pad_token_id, self.mask_token_id,
        ]
        # 拍号 tokens
        ids.extend(range(self.time_sig_offset_id, self.time_sig_offset_id + 5))
        # BPM tokens
        ids.extend(range(self.bpm_offset_id, self.bpm_offset_id + 4))
        return set(ids)


@dataclass
class BertPretrainConfig:
    """BERT 预训练超参数"""
    # 模型架构
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # MLM 参数
    mask_prob: float = 0.15       # 15% 的 note token 被选中
    mask_replace_prob: float = 0.8   # 其中 80% 替换为 [MASK]
    mask_random_prob: float = 0.1    # 10% 替换为随机 token
    # 剩余 10% 保持原样

    # 训练参数
    data_dir: str = "/path/to/data/npz"
    output_dir: str = "./checkpoints/music_bert_with_pair"
    num_epochs: int = 30
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 2048
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # 日志与保存
    log_every_n_steps: int = 50
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    test_split_ratio: float = 0.05
    random_seed: int = 42

    # DataLoader
    num_workers: int = 4
