"""
Token to MIDI Converter

将模型生成的token序列转换为MIDI文件的完整流程：
1. 提取元数据（拍号、BPM）
2. 按bar_token_id划分小节
3. 对每个小节分离双轨道并解码为pianoroll
4. 拼接所有小节
5. 转换为MIDI文件

Author: Refactored version
"""

import numpy as np
import torch
import pretty_midi
from typing import Union, Optional, Tuple, List
from my_tokenizer import PianoRollTokenizer


class Token2MIDI:
    """
    Token序列到MIDI的转换器

    处理流程：
    1. 从token序列提取元数据（拍号、BPM）
    2. 按bar_token_id划分小节
    3. 对每个小节：
       - 分离track0和track1的beat tokens
       - 使用tokenizer解码为pianoroll
       - 拼接成完整小节
    4. 合并所有小节
    5. 对齐双轨道时间长度
    6. 转换为MIDI文件

    参数：
        tokenizer: PianoRollTokenizer实例
        config: ModelConfig配置对象

    示例：
        >>> from my_tokenizer import PianoRollTokenizer
        >>> from config import ModelConfig
        >>>
        >>> config = ModelConfig()
        >>> tokenizer = PianoRollTokenizer(
        ...     patch_h=config.patch_h,
        ...     patch_w=config.patch_w,
        ...     pattern_num=config.pattern_num,
        ...     beats_length=config.beats_length
        ... )
        >>> converter = Token2MIDI(tokenizer, config)
        >>> converter.convert(token_sequence, "output.mid", tempo=120)
    """

    def __init__(self, tokenizer: PianoRollTokenizer, config):
        """
        初始化转换器

        Args:
            tokenizer: PianoRollTokenizer实例
            config: 配置对象，包含各种token ID定义
        """
        self.tokenizer = tokenizer

        # Token ID配置
        self.bar_token_id = config.bar_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.track0_start_id = config.track0_start_id
        self.track1_start_id = config.track1_start_id

        # 元数据配置
        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id

        # Patch配置
        self.patch_w = config.patch_w  # 每个beat的时间步长度

    def convert(
        self,
        token_sequence: Union[torch.Tensor, np.ndarray, list],
        output_path: str,
        tempo: Optional[int] = None,
        velocity: int = 64,
        merge_tracks: bool = False
    ) -> str:
        """
        将token序列转换为MIDI文件（主入口）

        Args:
            token_sequence: 完整的token序列（包含BOS/EOS）
            output_path: MIDI文件保存路径
            tempo: 节拍速度（BPM），如果为None则从序列中提取
            velocity: MIDI音符力度 (1-127)

        Returns:
            output_path: 保存的MIDI文件路径
        """
        # 转换为numpy array
        if isinstance(token_sequence, torch.Tensor):
            token_sequence = token_sequence.cpu().numpy()
        elif isinstance(token_sequence, list):
            token_sequence = np.array(token_sequence)

        print(f"原始token序列长度: {len(token_sequence)}")

        # 1. 提取元数据
        time_signature, bpm = self._extract_metadata(token_sequence)
        if tempo is None:
            tempo = bpm
        print(f"元数据 - 拍号: {time_signature}, BPM: {bpm} (使用tempo={tempo})")

        # 2. 提取音乐内容（去除BOS、元数据、EOS）
        music_tokens = self._extract_music_tokens(token_sequence)
        print(f"音乐内容token数量: {len(music_tokens)}")

        # 3. 按小节划分
        bars = self._split_by_bars(music_tokens)
        print(f"划分为 {len(bars)} 个小节")

        # 4. 处理每个小节，得到双轨道pianoroll
        track0_pianorolls = []
        track1_pianorolls = []

        for bar_idx, bar_tokens in enumerate(bars):
            track0_pr, track1_pr = self._process_bar(bar_tokens)
            # 对齐每个小节的双轨道长度
            track0_pr, track1_pr = self._align_tracks(track0_pr, track1_pr)
            track0_pianorolls.append(track0_pr)
            track1_pianorolls.append(track1_pr)

        # 5. 拼接所有小节（已在小节级别对齐，无需再次对齐）
        track0_full = np.concatenate(track0_pianorolls, axis=-1)  # (2, 88, total_time)
        track1_full = np.concatenate(track1_pianorolls, axis=-1)
        # 6. 合并双轨道为4通道pianoroll
        combined_pianoroll = np.concatenate([track0_full, track1_full], axis=0)  # (4, 88, time)
        combined_pianoroll = combined_pianoroll[:, ::-1, :].copy()
        if merge_tracks:
            # 如果需要合并为单轨道，可以在这里实现
            combined_pianoroll = np.maximum(track0_full, track1_full)
            self._pianoroll_to_midi_single_track(combined_pianoroll, output_path, tempo, velocity)
        else:
        # 8. 转换为MIDI
            self._pianoroll_to_midi(combined_pianoroll, output_path, tempo, velocity)

        return output_path

    def _extract_metadata(self, token_sequence: np.ndarray) -> Tuple[str, int]:
        """
        提取拍号和BPM

        序列格式: [BOS, time_sig, bpm, music_content..., EOS]

        Args:
            token_sequence: 完整token序列

        Returns:
            time_signature: 拍号字符串 (如 "4/4")
            bpm: 节拍速度
        """
        # 拍号映射 (根据config.py中的time_sig_offset_id)
        time_sig_map = {
            0: '4/4',
            1: '3/4',
            2: '2/4',
            3: '6/8',
            4: '2/2'
        }

        # BPM映射 (根据PianoDataset.py中的encode_bpm函数)
        bpm_map = {
            0: 80,   # 慢速 <90
            1: 120,  # 中速 90-200
            2: 220,  # 快速 >200
            3: 120   # 未知，默认120
        }

        # 提取拍号 (位置1)
        time_sig_token = int(token_sequence[1])
        time_sig_idx = time_sig_token - self.time_sig_offset_id
        time_signature = time_sig_map.get(time_sig_idx, '4/4')

        # 提取BPM (位置2)
        bpm_token = int(token_sequence[2])
        bpm_idx = bpm_token - self.bpm_offset_id
        bpm = bpm_map.get(bpm_idx, 120)

        return time_signature, bpm

    def _extract_music_tokens(self, token_sequence: np.ndarray) -> np.ndarray:
        """
        提取音乐内容token（去除BOS、元数据、EOS）

        Args:
            token_sequence: 完整token序列

        Returns:
            music_tokens: 纯音乐内容的token序列
        """
        # 去除BOS (位置0)、拍号 (位置1)、BPM (位置2)
        start_idx = 3

        # 查找EOS并去除
        end_idx = len(token_sequence)
        if token_sequence[-1] == self.eos_token_id:
            end_idx = -1

        return token_sequence[start_idx:end_idx]

    def _split_by_bars(self, music_tokens: np.ndarray) -> List[np.ndarray]:
        """
        按bar_token_id划分小节

        Args:
            music_tokens: 音乐内容token序列

        Returns:
            bars: 小节token列表，每个元素是一个小节的tokens（不包含bar_token_id本身）
        """
        # 找到所有bar token的位置
        bar_positions = np.where(music_tokens == self.bar_token_id)[0]

        bars = []
        for i in range(len(bar_positions)):
            start = bar_positions[i] + 1  # 跳过bar token本身
            end = bar_positions[i + 1] if i + 1 < len(bar_positions) else len(music_tokens)
            bar_tokens = music_tokens[start:end]
            if len(bar_tokens) > 0:  # 忽略空小节
                bars.append(bar_tokens)

        return bars

    def _process_bar(self, bar_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单个小节：分离双轨道并解码为pianoroll

        小节内格式：[track0_beat0, track1_beat0, track0_beat1, track1_beat1, ...]

        Args:
            bar_tokens: 单个小节的token序列

        Returns:
            track0_pianoroll: track0的pianoroll (2, 88, time)
            track1_pianoroll: track1的pianoroll (2, 88, time)
        """
        # 分离track0和track1的beats
        track0_beats, track1_beats = self._separate_tracks(bar_tokens)

        # 解码每个轨道
        track0_pianoroll = self._decode_track_beats(track0_beats, self.track0_start_id)
        track1_pianoroll = self._decode_track_beats(track1_beats, self.track1_start_id)

        return track0_pianoroll, track1_pianoroll

    def _separate_tracks(self, bar_tokens: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        从小节中分离track0和track1的beat tokens

        Args:
            bar_tokens: 小节内的所有tokens

        Returns:
            track0_beats: track0的beat token列表
            track1_beats: track1的beat token列表
        """
        # 找到所有track标记的位置
        track0_positions = np.where(bar_tokens == self.track0_start_id)[0]
        track1_positions = np.where(bar_tokens == self.track1_start_id)[0]

        # 合并所有track标记位置并排序
        all_positions = np.sort(np.concatenate([track0_positions, track1_positions]))

        track0_beats = []
        track1_beats = []

        for i in range(len(all_positions)):
            start = all_positions[i]
            end = all_positions[i + 1] if i + 1 < len(all_positions) else len(bar_tokens)
            beat_tokens = bar_tokens[start:end]

            # 判断是哪个轨道
            if bar_tokens[start] == self.track0_start_id:
                track0_beats.append(beat_tokens)
            elif bar_tokens[start] == self.track1_start_id:
                track1_beats.append(beat_tokens)

        return track0_beats, track1_beats

    def _decode_track_beats(self, beat_token_list: List[np.ndarray], track_id: int) -> np.ndarray:
        """
        解码单个轨道的所有beats为pianoroll

        Args:
            beat_token_list: beat token列表
            track_id: 轨道标记ID

        Returns:
            pianoroll: 解码后的pianoroll (2, 88, total_time)
        """
        beat_pianorolls = []

        for beat_tokens in beat_token_list:
            # 解压缩tokens为token矩阵
            token_matrix = self.tokenizer.decompress_tokens(beat_tokens, track_marker_id=track_id)

            # 转换为pianoroll
            pianoroll = self.tokenizer.patch_tokens_to_image(token_matrix)  # (2, 88, patch_w)

            beat_pianorolls.append(pianoroll)

        # 拼接所有beats
        if len(beat_pianorolls) > 0:
            full_pianoroll = np.concatenate(beat_pianorolls, axis=-1)
        else:
            # 空轨道
            full_pianoroll = np.zeros((2, 88, self.patch_w), dtype=np.float32)

        return full_pianoroll

    def _align_tracks(
        self,
        track0: np.ndarray,
        track1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对齐两个轨道的时间长度（padding到相同长度）

        Args:
            track0: track0的pianoroll (2, 88, t0)
            track1: track1的pianoroll (2, 88, t1)

        Returns:
            track0_aligned: 对齐后的track0
            track1_aligned: 对齐后的track1
        """
        t0 = track0.shape[-1]
        t1 = track1.shape[-1]

        if t0 == t1:
            return track0, track1

        max_len = max(t0, t1)

        # Padding较短的轨道
        if t0 < max_len:
            pad_width = ((0, 0), (0, 0), (0, max_len - t0))
            track0 = np.pad(track0, pad_width, mode='constant', constant_values=0)

        if t1 < max_len:
            pad_width = ((0, 0), (0, 0), (0, max_len - t1))
            track1 = np.pad(track1, pad_width, mode='constant', constant_values=0)

        return track0, track1

    def _pianoroll_to_midi(
        self,
        pianoroll: np.ndarray,
        output_path: str,
        tempo: int,
        velocity: int
    ):
        """
        将4通道pianoroll转换为MIDI文件

        Args:
            pianoroll: (4, 88, time) - [track0_sustain, track0_onset, track1_sustain, track1_onset]
            output_path: MIDI文件保存路径
            tempo: 节拍速度 (BPM)
            velocity: MIDI音符力度
        """
        # 创建MIDI对象
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # 计算每个时间单位的秒数（基于patch_w，通常是1/16音符）
        seconds_per_step = 60.0 / tempo / 4  # 假设patch_w=4对应1拍，每步1/16拍

        # 处理track0 (通常是高声部/旋律)
        track0_sustain = pianoroll[0]  # (88, time)
        track0_onset = pianoroll[1]
        piano_track0 = self._create_midi_track(
            track0_sustain,
            track0_onset,
            seconds_per_step,
            velocity,
            program=0  # Acoustic Grand Piano
        )
        midi.instruments.append(piano_track0)

        # 处理track1 (通常是低声部/伴奏)
        track1_sustain = pianoroll[2]
        track1_onset = pianoroll[3]
        piano_track1 = self._create_midi_track(
            track1_sustain,
            track1_onset,
            seconds_per_step,
            velocity,
            program=0
        )
        midi.instruments.append(piano_track1)

        # 保存MIDI文件
        midi.write(output_path)

        print(f"\nMIDI文件已保存到: {output_path}")
        print(f"总时长: {midi.get_end_time():.2f} 秒")
        print(f"Track0音符数: {len(piano_track0.notes)}")
        print(f"Track1音符数: {len(piano_track1.notes)}")

    def _pianoroll_to_midi_single_track(
        self,
        pianoroll: np.ndarray,
        output_path: str,
        tempo: int,
        velocity: int
    ):
        """
        将4通道pianoroll转换为MIDI文件

        Args:
            pianoroll: (4, 88, time) - [track0_sustain, track0_onset, track1_sustain, track1_onset]
            output_path: MIDI文件保存路径
            tempo: 节拍速度 (BPM)
            velocity: MIDI音符力度
        """
        # 创建MIDI对象
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # 计算每个时间单位的秒数（基于patch_w，通常是1/16音符）
        seconds_per_step = 60.0 / tempo / 4  # 假设patch_w=4对应1拍，每步1/16拍

        # 处理track0 (通常是高声部/旋律)
        track0_sustain = pianoroll[0]  # (88, time)
        track0_onset = pianoroll[1]
        piano_track0 = self._create_midi_track(
            track0_sustain,
            track0_onset,
            seconds_per_step,
            velocity,
            program=0  # Acoustic Grand Piano
        )
        midi.instruments.append(piano_track0)


        # 保存MIDI文件
        midi.write(output_path)

        print(f"\nMIDI文件已保存到: {output_path}")
        print(f"总时长: {midi.get_end_time():.2f} 秒")
        print(f"Track0音符数: {len(piano_track0.notes)}")

    def _create_midi_track(
        self,
        sustain_roll: np.ndarray,
        onset_roll: np.ndarray,
        seconds_per_step: float,
        velocity: int,
        program: int = 0
    ) -> pretty_midi.Instrument:
        """
        从sustain和onset pianoroll创建MIDI轨道

        Args:
            sustain_roll: (88, time) sustain通道
            onset_roll: (88, time) onset通道
            seconds_per_step: 每个时间步的秒数
            velocity: 音符力度
            program: MIDI乐器编号

        Returns:
            instrument: PrettyMIDI乐器对象
        """
        instrument = pretty_midi.Instrument(program=program)

        # 遍历每个音高
        for pitch_idx in range(88):
            pitch = pitch_idx + 21  # MIDI音高 (A0=21开始)

            # 找到所有onset位置
            onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]

            for onset_pos in onset_positions:
                # 找到音符结束位置（sustain结束）
                end_pos = onset_pos + 1

                # 继续向后查找直到sustain结束
                while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                    end_pos += 1

                # 转换为时间（秒）
                start_time = onset_pos * seconds_per_step
                end_time = end_pos * seconds_per_step

                # 创建MIDI音符
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

        return instrument
