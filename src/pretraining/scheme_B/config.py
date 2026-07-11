"""
Music BERT 配置

基于 no_pair_related 编码方案 (相对位置编码, vocab=184)
在原始音乐词表基础上增加 [MASK] token 用于 MLM 预训练
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """音乐 token 编码配置 (no_pair_related 方案)"""
    # Patch 编码参数
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 种三进制 pattern
    beats_length: int = 88    # 88 钢琴键

    # 原始词表: 0-183 (184 tokens)
    # patch value tokens: 0-80
    # relative position tokens: 81-168 (81 + relative_distance)
    # 特殊 tokens: 169-183
    original_vocab_size: int = 184

    # 特殊 token IDs
    empty_marker_id: int = 169    # 空beat标记
    end_marker_id: int = 170      # 非空beat结束标记
    bar_token_id: int = 171       # 小节线
    eos_token_id: int = 172
    bos_token_id: int = 173
    pad_token_id: int = 174
    time_sig_offset_id: int = 175  # 175-179: 5种拍号
    bpm_offset_id: int = 180       # 180-183: 4种速度

    # MLM 新增
    mask_token_id: int = 184      # [MASK] token
    vocab_size: int = 185         # 原始 184 + [MASK]

    # Note token 范围 (用于 MLM masking 判断)
    # 包括 patch values (0-80) 和 relative positions (81-168)
    note_token_min: int = 0
    note_token_max: int = 168

    @property
    def special_token_ids(self):
        """所有不应被 mask 的特殊 token"""
        ids = [
            self.empty_marker_id, self.end_marker_id,
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
    output_dir: str = "./checkpoints/music_bert"
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
