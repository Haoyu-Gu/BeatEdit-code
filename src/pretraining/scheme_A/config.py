"""
Music BERT 配置

基于 no_pair 编码方案 (绝对位置编码, vocab=185)
在原始音乐词表基础上增加 [MASK] token 用于 MLM 预训练

注: 原始 LLaMA 模型 vocab_size=268 (有保留位), 实际最大 token ID=184.
    BERT 使用紧凑词表: 0-184 + [MASK]=185, 共 186.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MusicTokenConfig:
    """音乐 token 编码配置 (no_pair 方案 - 绝对位置编码)"""
    # Patch 编码参数
    patch_h: int = 1
    patch_w: int = 4
    pattern_num: int = 81     # 3^4 = 81 种三进制 pattern
    beats_length: int = 88    # 88 钢琴键

    # 原始词表: 实际使用 0-184 (185 tokens)
    # patch value tokens: 0-80
    # absolute position tokens: 81-168 (81 + pitch_index)
    # (token 169 未使用)
    # 特殊 tokens: 170-184
    original_vocab_size: int = 185

    # 特殊 token IDs (no_pair 方案无 EMPTY/END marker)
    bar_token_id: int = 170
    eos_token_id: int = 171
    bos_token_id: int = 172
    pad_token_id: int = 173
    time_sig_offset_id: int = 174  # 174-178: 5种拍号
    bpm_offset_id: int = 179       # 179-182: 4种速度
    track0_start_id: int = 183     # Track 0 开始标记
    track1_start_id: int = 184     # Track 1 开始标记

    # MLM 新增
    mask_token_id: int = 185      # [MASK] token
    vocab_size: int = 186         # 实际 185 + [MASK]

    # Note token 范围 (用于 MLM masking 判断)
    # 包括 patch values (0-80) 和 absolute positions (81-168)
    note_token_min: int = 0
    note_token_max: int = 168

    @property
    def special_token_ids(self):
        """所有不应被 mask 的特殊 token"""
        ids = [
            self.bar_token_id, self.eos_token_id, self.bos_token_id,
            self.pad_token_id, self.mask_token_id,
            self.track0_start_id, self.track1_start_id,
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
    output_dir: str = "./checkpoints/music_bert_no_pair"
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
