# dataset.py - 修改后的完整代码

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pickle
from typing import List, Dict
from dataclasses import dataclass
from my_tokenizer import PianoRollTokenizer



def encode_bpm(bpm):
        if bpm is None:
            return 3  # UNK token
        bpm = int(bpm)
        if bpm < 90:
            return 0  # 慢速
        elif bpm <= 200:
            return 1  # 中速
        else:
            return 2  # 快速
        


class PianoDataset(Dataset):
    """支持长度感知的数据集"""

    def __init__(self, data_dir, config, cache_lengths=True, mode='train',
                 test_split_ratio=0.05, random_seed=42):
        """
        Args:
            data_dir: 数据目录
            config: 模型配置
            cache_lengths: 是否使用长度缓存
            mode: 'train' 或 'test'，决定使用训练集还是测试集
            test_split_ratio: 测试集划分比例（0-1之间）
            random_seed: 随机种子，用于可重复的数据集划分
        """
        self.root_dir = data_dir

        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.max_seq_len = config.train_cutoff_len

        self.pad_token = config.pad_token_id
        self.bos_token = config.bos_token_id
        self.eos_token = config.eos_token_id
        self.bar_token = config.bar_token_id

        self.empty_marker_id = config.empty_marker_id
        self.end_marker_id = config.end_marker_id

        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id

        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed

        self.config = config

        # 创建tokenizer实例
        self.tokenizer = PianoRollTokenizer(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            pattern_num=config.pattern_num,
            beats_length=config.beats_length,
        )

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        print(f"找到 {len(self.data_files)} 个有效的npz文件")

        # 预计算长度信息
        cache_file = os.path.join(data_dir, '.lengths_cache.pkl')

        if cache_lengths and os.path.exists(cache_file):
            print("加载长度缓存...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 验证patch参数是否匹配
            if (cache_data['patch_h'] != self.patch_h or 
                cache_data['patch_w'] != self.patch_w):
                raise ValueError(
                    f"缓存的patch参数({cache_data['patch_h']}x{cache_data['patch_w']}) "
                    f"与配置({self.patch_h}x{self.patch_w})不匹配，请重新运行precompute_lengths.py"
                )
            
            self.data_files = cache_data['data_files']
            self.file_lengths = cache_data['lengths']
            self.sorted_indices = cache_data['sorted_indices']
            
            print(f"加载 {len(self.data_files)} 个文件的长度信息")
            
        elif cache_lengths:
            raise FileNotFoundError(
                f"长度缓存不存在: {cache_file}\n"
                f"请先运行: python precompute_lengths.py"
            )
        else:
            # 不使用缓存，传统方式
            self.file_lengths = None
            self.sorted_indices = None
            print(f"找到 {len(self.data_files)} 个文件（未使用长度缓存）")

        # 划分训练集和测试集
        self._split_train_test()

    def _split_train_test(self):
        """根据mode参数划分训练集和测试集"""
        total_files = len(self.data_files)

        # 设置随机种子以确保可重复性
        rng = np.random.RandomState(self.random_seed)
    
        # 创建索引数组并打乱
        indices = np.arange(total_files)
        rng.shuffle(indices)  # 只影响这里的shuffle


        # 计算测试集大小
        test_size = int(total_files * self.test_split_ratio)
        train_size = total_files - test_size

        if self.config.min_length > 0:
            # 根据长度过滤样本
            filtered_indices = []
            for i in indices:
                length = self.file_lengths[i]
                if length > self.config.min_length:
                    filtered_indices.append(i)
            indices = np.array(filtered_indices)
            total_files = len(indices)
            test_size = int(total_files * self.test_split_ratio)
            train_size = total_files - test_size
            print(f"过滤后剩余 {total_files} 个文件")

        if self.mode == 'train':
            # 使用前train_size个样本作为训练集
            selected_indices = indices[:train_size]
            print(f"使用训练集: {len(selected_indices)} 个文件 ({train_size}/{total_files})")
        elif self.mode == 'test':
            # 使用后test_size个样本作为测试集
            selected_indices = indices[train_size:]
            print(f"使用测试集: {len(selected_indices)} 个文件 ({test_size}/{total_files})")
        elif self.mode == 'all':
            # 使用全部数据
            selected_indices = indices
            print(f"使用全部数据: {len(selected_indices)} 个文件 ({total_files}/{total_files})")
        elif self.mode == 'toy':
            # 使用全部数据
            selected_indices = indices[:4000]
            print(f"使用验证数据: {len(selected_indices)} 个文件 ({4000}/{total_files})")
        else:
            raise ValueError(f"mode必须是'train'或'test'或'all'，当前为: {self.mode}")

        # 更新data_files和相关索引
        self.data_files = [self.data_files[i] for i in selected_indices]

        # 如果使用了长度缓存，也需要更新相关信息
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in selected_indices]

            # 重新创建sorted_indices（在新的子集中的排序）
            self.sorted_indices = sorted(
                range(len(self.file_lengths)),
                key=lambda i: self.file_lengths[i]
            )

    def __len__(self):
        return len(self.data_files)
    
    def _process_measure_with_beat_interleaving(
        self,
        measure,
    ):
        """
        在每一拍级别交错处理高声部和低声部

        Args:
            measure: (4, 88, t) - 前2通道part0(高声部)，后2通道part1(低声部)
            tokenizer: PianoRollTokenizer实例
            timesteps_per_beat: 每拍的时间步数

        Returns:
            part0_beat_tokens: part0的beat级别token列表
            part1_beat_tokens: part1的beat级别token列表
        """
        part0_beat_tokens = []
        part1_beat_tokens = []

        t = measure.shape[2]
        beat_length = self.patch_w

        # 计算实际有多少拍
        num_beats = (t + beat_length - 1) // beat_length  # 向上取整

        for beat_idx in range(num_beats):
            # 计算当前拍的时间范围
            start_t = beat_idx * beat_length
            end_t = min(start_t + beat_length, t)

            # 如果最后一拍不够长，需要padding
            beat_measure = measure[:, :, start_t:end_t]
            current_length = end_t - start_t

            if current_length < beat_length:
                # Padding到完整的拍长度
                pad_width = ((0, 0), (0, 0), (0, beat_length - current_length))
                beat_measure = np.pad(beat_measure, pad_width, mode='constant', constant_values=0)

            # === 处理part0 (高声部) ===
            part0_beat = beat_measure[:2]  # (2, 88, beat_length)
            tokens_0 = self.tokenizer.image_to_patch_tokens(part0_beat, strict_mode=True)
            compressed_tokens_0 = self.tokenizer.compress_tokens(
                tokens_0, empty_marker_id=self.empty_marker_id, end_marker_id=self.end_marker_id)
            part0_beat_tokens.append(torch.tensor(compressed_tokens_0, dtype=torch.long))

            # === 处理part1 (低声部) ===
            part1_beat = beat_measure[2:]  # (2, 88, beat_length)
            tokens_1 = self.tokenizer.image_to_patch_tokens(part1_beat, strict_mode=True)
            compressed_tokens_1 = self.tokenizer.compress_tokens(
                tokens_1, empty_marker_id=self.empty_marker_id, end_marker_id=self.end_marker_id)
            part1_beat_tokens.append(torch.tensor(compressed_tokens_1, dtype=torch.long))

        return part0_beat_tokens, part1_beat_tokens
    

    def _interleave_tokens(self, part0_beats, part1_beats):
        """
        交错拼接 part0 和 part1 的 tokens

        Args:
            part0_beats: part0的beat tokens列表  [tensor0, tensor0, ...]
            part1_beats: part1的beat tokens列表

        Returns:
            交错拼接后的tensor
        """
        tokens = []
        for p0, p1 in zip(part0_beats, part1_beats):
            tokens.append(p0)
            tokens.append(p1)
        return torch.cat(tokens, dim=0)

    
    def __getitem__(self, idx):
        # 加载数据
        file_path = os.path.join(self.root_dir, self.data_files[idx])
        file_name = self.data_files[idx]
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()

        # 提取元数据
        time_sig_idx = metadata['time_signature_idx']
        if time_sig_idx == 9:
            time_sig_idx = 4  # 处理特殊拍号

        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        is_continuation = metadata.get('is_continuation', False)

        # 判断是否添加BOS（根据文件名）
        add_bos = True
        if '_' in file_name:
            suffix = file_name.split('_')[-1].replace('.npz', '')
            if suffix.isdigit() and suffix != '1':
                add_bos = False

        # 随机音高偏移（数据增强）
        pitch_shift = 0
        if np.random.random() < 0.7:
            pitch_shift = np.random.randint(-5, 6)

        # 处理所有measures，收集tokens
        measure_tokens = []
        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']
            measure = measure[:, ::-1, :].copy()
            # 应用音高偏移
            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            # 处理beat并交错
            part0_beats, part1_beats = self._process_measure_with_beat_interleaving(measure)
            beat_tokens = self._interleave_tokens(part0_beats, part1_beats)

            # 添加bar token和beat tokens
            measure_tokens.append(torch.tensor([self.bar_token], dtype=torch.long))
            measure_tokens.append(beat_tokens)

        # 构建完整序列
        tokens = []
        labels = []

        # 添加BOS
        if add_bos:
            tokens.append(torch.tensor([self.bos_token], dtype=torch.long))
            labels.append(torch.tensor([-100], dtype=torch.long))

        # 添加拍号和BPM
        time_sig_token = time_sig_idx + self.time_sig_offset_id
        bpm_token = encode_bpm(bpm_value) + self.bpm_offset_id
        tokens.append(torch.tensor([time_sig_token, bpm_token], dtype=torch.long))
        labels.append(torch.tensor([-100, -100], dtype=torch.long))

        # 添加音乐内容
        content = torch.cat(measure_tokens)
        tokens.append(content)
        labels.append(content)

        # 添加EOS
        if not is_continuation:
            tokens.append(torch.tensor([self.eos_token], dtype=torch.long))
            labels.append(torch.tensor([self.eos_token], dtype=torch.long))

        # 拼接所有tokens
        input_ids = torch.cat(tokens)
        labels = torch.cat(labels)

        # 超长截断：随机截取片段，前8%不计入损失
        seq_len = len(input_ids)

        if seq_len > self.max_seq_len :
            prob = np.random.random()
            if  prob < 0.25:
                # 从开头截断
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]
            elif prob < 0.5:
                # 从结尾截断
                input_ids = input_ids[-self.max_seq_len:]
                labels = labels[-self.max_seq_len:]

                # 前8%不计入损失
                ignore_len = int(self.max_seq_len * 0.08)
                labels[:ignore_len] = -100

            else:
                start_idx = np.random.randint(0, seq_len - self.max_seq_len + 1)
                input_ids = input_ids[start_idx:start_idx + self.max_seq_len]
                labels = labels[start_idx:start_idx + self.max_seq_len]

                # 前8%不计入损失
                ignore_len = int(self.max_seq_len * 0.08)
                labels[:ignore_len] = -100
            
        

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class BucketBatchSampler(Sampler):
    """长度感知的批采样器"""
    
    def __init__(self, dataset, batch_size=16, bucket_size=100, shuffle=True):
        """
        Args:
            dataset: PianoDataset实例
            batch_size: 实际训练的batch大小
            bucket_size: 每个长度bucket的大小
            shuffle: 是否随机化
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        if dataset.sorted_indices is None:
            raise ValueError("Dataset需要启用cache_lengths=True")
        
        self._create_buckets()
    
    def _create_buckets(self):
        """将相近长度的样本分组到buckets中"""
        self.buckets = []
        sorted_indices = self.dataset.sorted_indices
        
        # 将排序后的索引分割成buckets
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size]
            self.buckets.append(bucket)
        
        print(f"创建了 {len(self.buckets)} 个长度buckets")
    
    def __iter__(self):
        """生成batch索引"""
        # 随机打乱buckets的顺序
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))
        
        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()
            
            # 在bucket内部随机打乱
            if self.shuffle:
                np.random.shuffle(bucket)
            
            # 从bucket中生成batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:  # 确保不是空batch
                    yield batch
    
    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets) // self.batch_size


@dataclass
class DataCollatorForVariableLengthLM:
    """数据整理器，支持动态padding"""
    
    def __init__(self, config):
        self.pad_token_id = config.pad_token_id
        self.max_length = config.train_cutoff_len
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 获取批次中的最大长度
        max_len = min(
            max(len(feature["input_ids"]) for feature in features),
            self.max_length
        )
        
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            seq_len = len(input_ids)
            
            if seq_len < max_len:
                padding_len = max_len - seq_len
                
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_len,), self.pad_token_id, dtype=torch.long)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((padding_len,), -100, dtype=torch.long)
                ])
                
                attention_mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(seq_len, dtype=torch.long)
            
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["attention_mask"].append(attention_mask)
        
        batch = {k: torch.stack(v) for k, v in batch.items()}
        
        return batch