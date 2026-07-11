"""
Piano Roll Tokenizer - 钢琴卷帘编解码器

统一管理piano roll与token之间的编解码逻辑，包括：
- Patch-based tokenization (三进制编码)
- 绝对位置压缩编码
- 双通道piano roll重建

Author: Optimized from original implementation
"""

import numpy as np
import torch
from typing import Optional, Union


class PianoRollTokenizer:
    """
    钢琴卷帘(Piano Roll)的Token编解码器

    功能：
    1. 将双通道piano roll (sustain + onset) 转换为patch tokens
       (三态约定与论文一致: 0=silent, 1=onset, 2=sustain continuation)
    2. 使用绝对位置编码压缩token序列
    3. 支持完整的编码-解码循环

    参数：
        patch_h: patch高度（默认2，对应2个音高）
        patch_w: patch宽度（默认4，对应4个时间步）
        pattern_num: 相对位置标记的偏移量（默认81）
        beats_length: 每个beat的pitch数量（默认88键）
        beats_length: 图像高度，即钢琴键数（默认88）

    示例：
        >>> tokenizer = PianoRollTokenizer(patch_h=2, patch_w=4)
        >>> # 编码
        >>> image = np.random.randint(0, 2, (2, 88, 16))
        >>> compressed = tokenizer.encode(image)
        >>> # 解码
        >>> reconstructed = tokenizer.decode(compressed, num_beats=4)
    """

    def __init__(
        self,
        patch_h: int = 1,
        patch_w: int = 4,
        pattern_num: int = 81,
        beats_length: int = 88,
        use_velocities: bool = False,
        velocity_offset: Optional[int] = None,
        num_velocities: Optional[int] = None,
    ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.pattern_num = pattern_num
        self.beats_length = beats_length
        self.use_velocities = use_velocities

        # Velocity相关参数
        if use_velocities:
            if velocity_offset is None:
                raise ValueError("velocity_offset must be provided when use_velocities=True")
            if num_velocities is None:
                raise ValueError("num_velocities must be provided when use_velocities=True")
            self.velocity_offset = velocity_offset
            self.num_velocities = num_velocities
        else:
            self.velocity_offset = velocity_offset
            self.num_velocities = num_velocities

        # 预计算patch相关参数
        self.patch_size = patch_h * patch_w
        self.powers_3 = 3 ** np.arange(self.patch_size - 1, -1, -1)

        # 特殊token替换规则（用于strict模式）
        self.special_token_ids = [
            26, 24, 34, 62, 47, 19, 29, 38, 74, 60,
            56, 7, 21, 65, 23, 22, 20, 25, 61, 11,
            6, 18, 2, 33, 8
        ]
        self.replacement_ids = [0, 53, 5, 80, 45]

    def quantize_velocity(self, velocity: Union[np.ndarray, int, float]) -> Union[np.ndarray, int]:
        """
        将1-127的velocity量化到1-num_velocities范围

        Args:
            velocity: 原始velocity值 (1-127)，可以是标量或数组

        Returns:
            量化后的velocity (1-num_velocities)
            0值保持为0（表示无音符）
        """
        if self.num_velocities is None:
            raise ValueError("num_velocities not set")

        if isinstance(velocity, np.ndarray):
            result = np.zeros_like(velocity, dtype=np.int64)
            non_zero_mask = velocity > 0
            # 将1-127映射到1-num_velocities
            # 使用线性量化: q = ceil((v / 127) * num_velocities)
            result[non_zero_mask] = np.ceil(
                velocity[non_zero_mask] / 127.0 * self.num_velocities
            ).astype(np.int64)
            # 确保范围在1-num_velocities
            result[non_zero_mask] = np.clip(result[non_zero_mask], 1, self.num_velocities)
            return result
        else:
            if velocity == 0:
                return 0
            q = int(np.ceil(velocity / 127.0 * self.num_velocities))
            return max(1, min(self.num_velocities, q))

    def dequantize_velocity(self, quantized: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        将量化的velocity (1-num_velocities) 还原到1-127范围

        Args:
            quantized: 量化后的velocity (1-num_velocities)

        Returns:
            还原后的velocity (1-127)
            0值保持为0（表示无音符）
        """
        if self.num_velocities is None:
            raise ValueError("num_velocities not set")

        if isinstance(quantized, np.ndarray):
            result = np.zeros_like(quantized, dtype=np.int64)
            non_zero_mask = quantized > 0
            # 反量化: v = round((q / num_velocities) * 127)
            result[non_zero_mask] = np.round(
                quantized[non_zero_mask] / self.num_velocities * 127.0
            ).astype(np.int64)
            # 确保范围在1-127
            result[non_zero_mask] = np.clip(result[non_zero_mask], 1, 127)
            return result
        else:
            if quantized == 0:
                return 0
            v = int(np.round(quantized / self.num_velocities * 127.0))
            return max(1, min(127, v))

    def encode(
        self,
        image: Union[np.ndarray, torch.Tensor],
        empty_marker_id: int,
        end_marker_id: int,
        use_strict_mode: bool = True
    ) -> np.ndarray:
        """
        完整编码流程：piano roll → compressed tokens

        Args:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID
            use_strict_mode: 是否使用严格模式（替换特殊token）

        Returns:
            compressed_sequence: 一维压缩token序列
        """
        tokens = self.image_to_patch_tokens(image, strict_mode=use_strict_mode)
        compressed = self.compress_tokens(tokens, empty_marker_id=empty_marker_id, end_marker_id=end_marker_id)
        return compressed

    def decode(
        self,
        compressed_sequence: Union[np.ndarray, list],
        empty_marker_id: int,
        end_marker_id: int
    ) -> np.ndarray:
        """
        完整解码流程：compressed tokens → piano roll

        Args:
            compressed_sequence: 压缩的token序列
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID

        Returns:
            image: shape (2, 88, t) 的双通道piano roll
        """
        tokens = self.decompress_tokens(compressed_sequence, empty_marker_id=empty_marker_id, end_marker_id=end_marker_id)
        image = self.patch_tokens_to_image(tokens)
        return image

    # ==================== 编码相关方法 ====================

    def image_to_patch_tokens(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True
    ) -> np.ndarray:
        """
        将双通道piano roll转换为patch tokens（三进制编码）

        Args:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
            strict_mode: 是否替换特殊token

        Returns:
            tokens: shape (num_time_patches, num_pitch_patches) 的token矩阵

        编码规则：
            0: 无音符 (sustain=0, onset=0)
            1: onset (sustain=1, onset=1)
            2: 只有sustain延续 (sustain=1, onset=0)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 确保输入是双通道
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain_channel = image[0].copy()  # shape: (88, t)
        onset_channel = image[1].copy()    # shape: (88, t)

        # onset只能出现在sustain为1的地方
        onset_channel[sustain_channel == 0] = 0

        beats_length, img_w = sustain_channel.shape

        # 处理宽度padding（确保可以整除patch_w）
        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain_channel = np.pad(
                sustain_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            onset_channel = np.pad(
                onset_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            img_w = sustain_channel.shape[1]

        num_patch_rows = beats_length // self.patch_h
        num_patch_cols = img_w // self.patch_w

        # 重塑为patches
        sustain_patches = self._reshape_to_patches(sustain_channel, num_patch_rows, num_patch_cols)
        onset_patches = self._reshape_to_patches(onset_channel, num_patch_rows, num_patch_cols)

        # 组合成三进制编码
        combined_patches = 2 * sustain_patches.astype(np.int64) - onset_patches.astype(np.int64)

        # 使用三进制计算token值
        tokens = np.dot(combined_patches, self.powers_3)

        # 处理特殊token（strict模式）
        if strict_mode:
            tokens = self._replace_special_tokens(tokens)

        return tokens

    def _reshape_to_patches(
        self,
        channel: np.ndarray,
        num_patch_rows: int,
        num_patch_cols: int
    ) -> np.ndarray:
        """将通道重塑为patches"""
        patches = channel.reshape(num_patch_rows, self.patch_h, num_patch_cols, self.patch_w)
        patches = patches.transpose(2, 0, 1, 3)  # (cols, rows, h, w)
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_size)
        return patches

    def _replace_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """随机替换特殊token"""
        mask = np.isin(tokens, self.special_token_ids)
        if np.any(mask):
            num_replacements = np.sum(mask)
            random_replacements = np.random.choice(self.replacement_ids, size=num_replacements)
            tokens = tokens.copy()
            tokens[mask] = random_replacements
        return tokens

    def image_patch_tokens_velocity(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True
    ) -> np.ndarray:
        """
        将双通道piano roll转换为带velocity的patch tokens

        Args:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain (0-127 velocity), ch1: onset (0-127 velocity)
                   velocity值范围为0-127
            strict_mode: 是否替换特殊token

        Returns:
            tokens: shape (2, num_time_patches, num_pitch_patches) 的双通道token矩阵
                    ch0: 三进制编码的音符pattern
                    ch1: 量化后的力度信息 (0-num_velocities)，0表示无音符，1-num_velocities表示量化后的velocity

        编码规则：
            channel 0 (pattern):
                0: 无音符 (sustain=0, onset=0)
                1: onset (sustain>0, onset>0)
                2: 只有sustain延续 (sustain>0, onset=0)
            channel 1 (velocity):
                0: 无音符
                1-num_velocities: patch内非零值的平均velocity量化后的值
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 确保输入是双通道
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain_channel = image[0].copy()  # shape: (88, t), velocity values 0-127
        onset_channel = image[1].copy()    # shape: (88, t), velocity values 0-127

        # onset只能出现在sustain为非零的地方
        onset_channel[sustain_channel == 0] = 0

        # 创建二值mask用于pattern编码
        sustain_binary = (sustain_channel > 0).astype(np.int64)
        onset_binary = (onset_channel > 0).astype(np.int64)

        beats_length, img_w = sustain_channel.shape

        # 处理宽度padding（确保可以整除patch_w）
        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain_binary = np.pad(
                sustain_binary,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            onset_binary = np.pad(
                onset_binary,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            sustain_channel = np.pad(
                sustain_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            img_w = sustain_binary.shape[1]

        num_patch_rows = beats_length // self.patch_h
        num_patch_cols = img_w // self.patch_w

        # 重塑为patches (用于pattern编码)
        sustain_patches = self._reshape_to_patches(sustain_binary, num_patch_rows, num_patch_cols)
        onset_patches = self._reshape_to_patches(onset_binary, num_patch_rows, num_patch_cols)

        # 组合成三进制编码
        combined_patches = 2 * sustain_patches.astype(np.int64) - onset_patches.astype(np.int64)

        # 使用三进制计算pattern token值
        pattern_tokens = np.dot(combined_patches, self.powers_3)

        # 处理特殊token（strict模式）
        if strict_mode:
            pattern_tokens = self._replace_special_tokens(pattern_tokens)

        # 计算velocity channel
        # 重塑sustain_channel为patches以计算平均velocity
        sustain_velocity_patches = self._reshape_to_patches(sustain_channel, num_patch_rows, num_patch_cols)

        # 计算每个patch内非零值的平均velocity
        velocity_tokens = np.zeros((num_patch_cols, num_patch_rows), dtype=np.int64)

        for col in range(num_patch_cols):
            for row in range(num_patch_rows):
                patch_values = sustain_velocity_patches[col, row]  # shape: (patch_size,)
                non_zero_values = patch_values[patch_values > 0]
                if len(non_zero_values) > 0:
                    # 计算平均值并四舍五入
                    avg_velocity = np.round(np.mean(non_zero_values)).astype(np.int64)
                    # 量化velocity到1-num_velocities范围
                    velocity_tokens[col, row] = self.quantize_velocity(avg_velocity)
                else:
                    velocity_tokens[col, row] = 0

        # 组合成双通道输出: (2, num_time_patches, num_pitch_patches)
        tokens = np.stack([pattern_tokens, velocity_tokens], axis=0)

        return tokens

    def compress_tokens(
        self,
        token_matrix: np.ndarray,
        empty_marker_id: int,
        end_marker_id: int
    ) -> np.ndarray:
        """
        使用相对位置编码压缩token序列

        Args:
            token_matrix: shape (num_beats, beats_length) 的token矩阵
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID

        Returns:
            compressed_sequence: 压缩后的一维序列

        编码格式：
            非空: [81+rel_pos0, val0, 81+rel_pos1, val1, ..., end_marker]
                  - 第一个位置：相对于0（即绝对位置）
                  - 后续位置：相对于上一个非零位置
            空:   [empty_marker]
        """
        compressed_sequences = []

        for beat_tokens in token_matrix:
            # 找到所有非零token的位置
            non_zero_indices = np.where(beat_tokens != 0)[0]

            if len(non_zero_indices) == 0:
                # 空beat
                compressed = [empty_marker_id]
            else:
                compressed = []
                prev_pos = 0

                for idx in non_zero_indices:
                    relative_pos = idx - prev_pos
                    position_marker = self.pattern_num + relative_pos
                    token_value = int(beat_tokens[idx])
                    compressed.extend([position_marker, token_value])
                    prev_pos = idx

                # 添加end_marker
                compressed.append(end_marker_id)

            compressed_sequences.append(np.array(compressed, dtype=np.int64))

        # 连接所有beats
        flattened_sequence = np.concatenate(compressed_sequences)

        return flattened_sequence

    def compress_tokens_velocity(
        self,
        token_matrix: np.ndarray,
        empty_marker_id: int,
        end_marker_id: int
    ) -> np.ndarray:
        """
        使用相对位置编码压缩带velocity的token序列

        Args:
            token_matrix: shape (2, num_beats, beats_length) 的双通道token矩阵
                          ch0: pattern tokens
                          ch1: velocity tokens (0-num_velocities)
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID

        Returns:
            compressed_sequence: 压缩后的一维序列

        编码格式：
            非空: [81+rel_pos0, token0, velocity0+offset, 81+rel_pos1, token1, velocity1+offset, ..., end_marker]
            空:   [empty_marker]
        """
        assert token_matrix.shape[0] == 2, f"Expected 2 channels, got {token_matrix.shape[0]}"

        pattern_matrix = token_matrix[0]   # shape: (num_beats, beats_length)
        velocity_matrix = token_matrix[1]  # shape: (num_beats, beats_length)

        compressed_sequences = []

        for beat_idx in range(pattern_matrix.shape[0]):
            beat_tokens = pattern_matrix[beat_idx]
            beat_velocities = velocity_matrix[beat_idx]

            # 找到所有非零token的位置
            non_zero_indices = np.where(beat_tokens != 0)[0]

            if len(non_zero_indices) == 0:
                # 空beat
                compressed = [empty_marker_id]
            else:
                compressed = []
                prev_pos = 0

                # 添加 [相对位置, token值, velocity值] 三元组
                for idx in non_zero_indices:
                    relative_pos = idx - prev_pos
                    position_marker = self.pattern_num + relative_pos
                    token_value = beat_tokens[idx]
                    velocity_value = beat_velocities[idx] + self.velocity_offset

                    compressed.extend([position_marker, token_value, velocity_value])
                    prev_pos = idx

                # 添加end_marker
                compressed.append(end_marker_id)

            compressed_sequences.append(np.array(compressed, dtype=np.int64))

        # 连接所有beats
        flattened_sequence = np.concatenate(compressed_sequences)

        return flattened_sequence

    # ==================== 解码相关方法 ====================

    def decompress_tokens(
        self,
        compressed_sequence: Union[np.ndarray, list],
        empty_marker_id: int,
        end_marker_id: int
    ) -> np.ndarray:
        """
        解压缩token序列（相对位置编码 → token矩阵）

        Args:
            compressed_sequence: 压缩的token序列
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID

        Returns:
            decompressed_beats: shape (num_beats, beats_length) 的token矩阵
        """
        if isinstance(compressed_sequence, list):
            compressed_sequence = np.array(compressed_sequence, dtype=np.int64)

        decompressed_beats = []
        i = 0

        while i < len(compressed_sequence):
            current_token = compressed_sequence[i]

            if current_token == empty_marker_id:
                # 空beat
                decompressed_beats.append(np.zeros(self.beats_length, dtype=np.int64))
                i += 1

            elif self.pattern_num <= current_token < empty_marker_id:
                # 非空beat - 当前token是第一个位置标记
                beat = np.zeros(self.beats_length, dtype=np.int64)
                current_pos = 0

                while i < len(compressed_sequence):
                    token = compressed_sequence[i]

                    if token == end_marker_id:
                        # beat结束
                        i += 1
                        break

                    if token == empty_marker_id:
                        # 下一个beat的开始（不消耗）
                        break

                    # 位置标记
                    if self.pattern_num <= token < empty_marker_id:
                        relative_pos = token - self.pattern_num
                        current_pos = current_pos + relative_pos
                        i += 1

                        if i >= len(compressed_sequence):
                            break

                        # 读取token值
                        token_value = compressed_sequence[i]
                        i += 1

                        # 填充token
                        if 0 <= current_pos < self.beats_length:
                            beat[current_pos] = token_value
                    else:
                        # 遇到非预期token，跳过
                        i += 1
                        break

                decompressed_beats.append(beat)

            else:
                # 跳过非预期token
                i += 1

        if len(decompressed_beats) == 0:
            return np.zeros((0, self.beats_length), dtype=np.int64)

        decompressed_sequence = np.stack(decompressed_beats, axis=0)

        return decompressed_sequence

    def decompress_tokens_velocity(
        self,
        compressed_sequence: Union[np.ndarray, list],
        empty_marker_id: int,
        end_marker_id: int
    ) -> np.ndarray:
        """
        解压缩带velocity的token序列（相对位置编码 → 双通道token矩阵）

        Args:
            compressed_sequence: 压缩的token序列
                格式: [81+rel_pos0, token0, velocity0+offset, ..., end_marker] 或 [empty_marker]
            empty_marker_id: 空beat标记ID
            end_marker_id: 非空beat结束标记ID

        Returns:
            decompressed_beats: shape (2, num_beats, beats_length) 的双通道token矩阵
                                ch0: pattern tokens
                                ch1: velocity tokens (0-num_velocities)，已减去velocity_offset
        """
        if isinstance(compressed_sequence, list):
            compressed_sequence = np.array(compressed_sequence, dtype=np.int64)

        decompressed_pattern_beats = []
        decompressed_velocity_beats = []
        i = 0

        while i < len(compressed_sequence):
            current_token = compressed_sequence[i]

            if current_token == empty_marker_id:
                # 空beat
                decompressed_pattern_beats.append(np.zeros(self.beats_length, dtype=np.int64))
                decompressed_velocity_beats.append(np.zeros(self.beats_length, dtype=np.int64))
                i += 1

            elif self.pattern_num <= current_token < empty_marker_id:
                # 非空beat
                pattern_beat = np.zeros(self.beats_length, dtype=np.int64)
                velocity_beat = np.zeros(self.beats_length, dtype=np.int64)
                current_pos = 0

                while i < len(compressed_sequence):
                    token = compressed_sequence[i]

                    if token == end_marker_id:
                        i += 1
                        break

                    if token == empty_marker_id:
                        break

                    # 位置标记
                    if self.pattern_num <= token < empty_marker_id:
                        relative_pos = token - self.pattern_num
                        current_pos = current_pos + relative_pos
                        i += 1

                        if i >= len(compressed_sequence):
                            break

                        # 读取token值
                        token_value = compressed_sequence[i]
                        i += 1

                        if i >= len(compressed_sequence):
                            break

                        # 读取velocity值
                        velocity_value = compressed_sequence[i]
                        i += 1

                        # 填充
                        if 0 <= current_pos < self.beats_length:
                            pattern_beat[current_pos] = token_value
                            velocity_beat[current_pos] = velocity_value - self.velocity_offset
                    else:
                        i += 1
                        break

                decompressed_pattern_beats.append(pattern_beat)
                decompressed_velocity_beats.append(velocity_beat)

            else:
                i += 1

        if len(decompressed_pattern_beats) == 0:
            return np.zeros((2, 0, self.beats_length), dtype=np.int64)

        pattern_sequence = np.stack(decompressed_pattern_beats, axis=0)
        velocity_sequence = np.stack(decompressed_velocity_beats, axis=0)

        # 组合成双通道输出: (2, num_beats, beats_length)
        decompressed_sequence = np.stack([pattern_sequence, velocity_sequence], axis=0)

        return decompressed_sequence

    def patch_tokens_to_image(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """
        从tokens重建双通道piano roll

        Args:
            tokens: shape (num_time_patches, num_pitch_patches) 的token矩阵

        Returns:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
        """
        num_patch_cols, num_patch_rows = tokens.shape

        # 解码tokens为三进制表示
        combined_patches = np.zeros(
            (num_patch_cols, num_patch_rows, self.patch_size),
            dtype=np.int64
        )
        temp_tokens = tokens.copy()

        for i in range(self.patch_size):
            combined_patches[:, :, i] = temp_tokens // self.powers_3[i]
            temp_tokens = temp_tokens % self.powers_3[i]

        # 从三进制值恢复双通道
        # 0 -> sustain=0, onset=0
        # 1 -> onset (sustain=1, onset=1)
        # 2 -> sustain延续 (sustain=1, onset=0)
        sustain_patches = (combined_patches >= 1).astype(np.float32)
        onset_patches = (combined_patches == 1).astype(np.float32)

        # 重建图像
        sustain_channel = self._patches_to_channel(
            sustain_patches,
            num_patch_cols,
            num_patch_rows
        )
        onset_channel = self._patches_to_channel(
            onset_patches,
            num_patch_cols,
            num_patch_rows
        )

        # 组合成双通道
        image = np.stack([sustain_channel, onset_channel], axis=0)
        return image

    def patch_tokens_to_image_velocity(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """
        从带velocity的tokens重建双通道piano roll

        Args:
            tokens: shape (2, num_time_patches, num_pitch_patches) 的双通道token矩阵
                    ch0: 三进制编码的音符pattern
                    ch1: 量化后的力度信息 (0-num_velocities)

        Returns:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain (velocity值), ch1: onset (velocity值)
                   velocity值范围为0-127（经过反量化）
        """
        assert tokens.shape[0] == 2, f"Expected 2 channels, got {tokens.shape[0]}"

        pattern_tokens = tokens[0]  # shape: (num_time_patches, num_pitch_patches)
        velocity_tokens = tokens[1]  # shape: (num_time_patches, num_pitch_patches)

        num_patch_cols, num_patch_rows = pattern_tokens.shape

        # 解码pattern tokens为三进制表示
        combined_patches = np.zeros(
            (num_patch_cols, num_patch_rows, self.patch_size),
            dtype=np.int64
        )
        temp_tokens = pattern_tokens.copy()

        for i in range(self.patch_size):
            combined_patches[:, :, i] = temp_tokens // self.powers_3[i]
            temp_tokens = temp_tokens % self.powers_3[i]

        # 从三进制值恢复双通道的二值mask
        # 0 -> sustain=0, onset=0
        # 1 -> onset (sustain=1, onset=1)
        # 2 -> sustain延续 (sustain=1, onset=0)
        sustain_mask = (combined_patches >= 1).astype(np.float32)
        onset_mask = (combined_patches == 1).astype(np.float32)

        # 反量化velocity: 从1-num_velocities还原到1-127
        dequantized_velocity = self.dequantize_velocity(velocity_tokens)

        # 将velocity应用到每个patch
        # dequantized_velocity shape: (num_patch_cols, num_patch_rows)
        # 需要扩展到 (num_patch_cols, num_patch_rows, patch_size)
        velocity_expanded = np.repeat(
            dequantized_velocity[:, :, np.newaxis],
            self.patch_size,
            axis=2
        ).astype(np.float32)

        # 应用velocity到sustain和onset mask
        sustain_patches = sustain_mask * velocity_expanded
        onset_patches = onset_mask * velocity_expanded

        # 重建图像
        sustain_channel = self._patches_to_channel(
            sustain_patches,
            num_patch_cols,
            num_patch_rows
        )
        onset_channel = self._patches_to_channel(
            onset_patches,
            num_patch_cols,
            num_patch_rows
        )

        # 组合成双通道
        image = np.stack([sustain_channel, onset_channel], axis=0)
        return image

    def _patches_to_channel(
        self,
        patches: np.ndarray,
        num_patch_cols: int,
        num_patch_rows: int
    ) -> np.ndarray:
        """将patches重建为通道"""
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_h, self.patch_w)
        patches = patches.transpose(1, 2, 0, 3)  # (rows, h, cols, w)
        channel = patches.reshape(self.beats_length, num_patch_cols * self.patch_w)
        return channel

    # ==================== 工具方法 ====================

    def get_config(self) -> dict:
        """返回配置字典"""
        return {
            'patch_h': self.patch_h,
            'patch_w': self.patch_w,
            'pattern_num': self.pattern_num,
            'beats_length': self.beats_length,
        }

    def __repr__(self) -> str:
        return (
            f"PianoRollTokenizer("
            f"patch_size={self.patch_h}x{self.patch_w}, "
            f"beats_length={self.beats_length}, "
            f"pattern_num={self.pattern_num})"
        )


