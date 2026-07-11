"""
FELIX-Music Datasets and Collators.

FELIXTaggerDataset: produces (input_ids, attention_mask, labels)
FELIXInserterDataset: produces (skeleton_ids, attention_mask, mask_positions, mask_targets)
Collators: dynamic padding
BucketBatchSampler: length-aware batching
"""

import os
import random
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict

from configs.config import (
    EMPTY_MARKER, SPLIT_0, SPLIT_1, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, BPM_OFFSET, LABEL_PAD, LABEL_KEEP, DATA_DIR,
)
from data.tokenizer import create_tokenizer
from data.sequence_parser import parse_sequence, separate_tracks, rebuild_interleaved
from data.perturbation import perturb_accompaniment
from data.label_extractor import extract_token_labels
from data.skeleton_builder import build_skeleton


def encode_bpm(bpm):
    """Quantize BPM to 4 categories."""
    if bpm is None:
        return 3
    bpm = int(bpm)
    if bpm < 90:
        return 0
    elif bpm <= 200:
        return 1
    else:
        return 2


class FELIXBaseDataset(Dataset):
    """Base dataset with shared tokenization logic."""

    def __init__(
        self,
        file_list: List[str],
        data_dir: str = DATA_DIR,
        max_len: int = 2048,
        pitch_shift_augment: bool = False,
        level_weights: tuple = (30, 30, 25, 15),
    ):
        self.file_list = file_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.pitch_shift_augment = pitch_shift_augment
        self.level_weights = level_weights
        self.tokenizer = create_tokenizer()

    def __len__(self):
        return len(self.file_list)

    def _tokenize_npz(self, idx):
        """Encode an npz file to absolute_bundled token sequence."""
        file_name = self.file_list[idx]
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            file_path = os.path.join(self.data_dir, file_name)

        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()

        time_sig_idx = metadata['time_signature_idx']
        if time_sig_idx == 9:
            time_sig_idx = 4

        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        is_continuation = metadata.get('is_continuation', False)

        add_bos = True
        if '_' in file_name:
            suffix = os.path.basename(file_name).split('_')[-1].replace('.npz', '')
            if suffix.isdigit() and suffix != '1':
                add_bos = False

        pitch_shift = 0
        if self.pitch_shift_augment and random.random() < 0.7:
            pitch_shift = random.randint(-5, 5)

        all_tokens = []

        if add_bos:
            all_tokens.append(BOS_TOKEN)
        all_tokens.append(TIME_SIG_OFFSET + time_sig_idx)
        all_tokens.append(BPM_OFFSET + encode_bpm(bpm_value))

        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']
            measure = measure[:, ::-1, :].copy()

            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            t = measure.shape[2]
            beat_length = 4
            num_beats = (t + beat_length - 1) // beat_length

            all_tokens.append(BAR_TOKEN)

            for beat_idx in range(num_beats):
                start_t = beat_idx * beat_length
                end_t = min(start_t + beat_length, t)
                beat_measure = measure[:, :, start_t:end_t]

                if end_t - start_t < beat_length:
                    pad_w = beat_length - (end_t - start_t)
                    beat_measure = np.pad(
                        beat_measure, ((0, 0), (0, 0), (0, pad_w)),
                        mode='constant', constant_values=0
                    )

                p0 = beat_measure[:2]
                tok0 = self.tokenizer.image_to_patch_tokens(p0, strict_mode=True)
                comp0 = self.tokenizer.compress_tokens(
                    tok0, split_marker_id=SPLIT_0, empty_marker_id=EMPTY_MARKER,
                )
                all_tokens.extend(comp0.tolist())

                p1 = beat_measure[2:]
                tok1 = self.tokenizer.image_to_patch_tokens(p1, strict_mode=True)
                comp1 = self.tokenizer.compress_tokens(
                    tok1, split_marker_id=SPLIT_1, empty_marker_id=EMPTY_MARKER,
                )
                all_tokens.extend(comp1.tolist())

        if not is_continuation:
            all_tokens.append(EOS_TOKEN)

        return all_tokens

    def _truncate(self, tokens):
        """Random truncation for sequences exceeding max_len."""
        if len(tokens) <= self.max_len:
            return tokens
        prob = random.random()
        if prob < 0.33:
            return tokens[:self.max_len]
        elif prob < 0.66:
            return tokens[-self.max_len:]
        else:
            start = random.randint(0, len(tokens) - self.max_len)
            return tokens[start:start + self.max_len]

    def _prepare_felix_data(self, idx):
        """
        Common data preparation: tokenize, perturb, extract token-level labels.

        Returns dict with source_tokens, target_tokens, labels, targets, or None.
        """
        try:
            target_tokens = self._tokenize_npz(idx)
            target_tokens = self._truncate(target_tokens)

            parsed_info = parse_sequence(target_tokens)
            melody_beats, accomp_beats = separate_tracks(parsed_info)

            if len(accomp_beats) == 0:
                return None

            original_accomp = copy.deepcopy(accomp_beats)

            perturbed_accomp, level, changed_mask = perturb_accompaniment(
                accomp_beats, level_weights=self.level_weights
            )

            source_tokens = rebuild_interleaved(melody_beats, perturbed_accomp, parsed_info)

            # Extract token-level labels
            labels, targets = extract_token_labels(source_tokens, target_tokens)

            return {
                'source_tokens': source_tokens,
                'target_tokens': target_tokens,
                'labels': labels,
                'targets': targets,
                'level': level,
            }
        except Exception:
            return None


class FELIXTaggerDataset(FELIXBaseDataset):
    """
    Dataset for FELIX Tagger model.

    Each sample produces:
    - input_ids: source token sequence (melody + perturbed accomp)
    - attention_mask: 1 for real tokens, 0 for padding
    - labels: per-token FELIX labels (same length as input_ids)
    """

    def __getitem__(self, idx):
        data = self._prepare_felix_data(idx)

        if data is None:
            return self._fallback_item(idx)

        source_tokens = data['source_tokens']
        labels = data['labels']

        return {
            'input_ids': torch.tensor(source_tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(source_tokens), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'length': len(source_tokens),
        }

    def _fallback_item(self, idx):
        try:
            token_sequence = self._tokenize_npz(idx)
            token_sequence = self._truncate(token_sequence)
        except Exception:
            token_sequence = [BOS_TOKEN, TIME_SIG_OFFSET, BPM_OFFSET, BAR_TOKEN, EMPTY_MARKER, EOS_TOKEN]

        labels = [LABEL_KEEP] * len(token_sequence)
        return {
            'input_ids': torch.tensor(token_sequence, dtype=torch.long),
            'attention_mask': torch.ones(len(token_sequence), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'length': len(token_sequence),
        }


class FELIXInserterDataset(FELIXBaseDataset):
    """
    Dataset for FELIX Inserter model.

    Each sample produces:
    - skeleton_ids: token sequence with MASKs at positions to fill
    - attention_mask: 1 for real tokens, 0 for padding
    - mask_positions: indices into skeleton_ids where MASKs are
    - mask_targets: target token IDs for each MASK position
    """

    def __getitem__(self, idx):
        data = self._prepare_felix_data(idx)

        if data is None:
            return self._fallback_item(idx)

        skeleton_result = build_skeleton(
            data['source_tokens'], data['labels'], data['targets']
        )

        skeleton_tokens = skeleton_result['skeleton_tokens']
        mask_positions = skeleton_result['mask_positions']
        mask_targets = skeleton_result['mask_targets']

        return {
            'skeleton_ids': torch.tensor(skeleton_tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(skeleton_tokens), dtype=torch.long),
            'mask_positions': torch.tensor(mask_positions, dtype=torch.long) if mask_positions else torch.zeros(0, dtype=torch.long),
            'mask_targets': torch.tensor(mask_targets, dtype=torch.long) if mask_targets else torch.zeros(0, dtype=torch.long),
            'length': len(skeleton_tokens),
            'num_masks': len(mask_positions),
        }

    def _fallback_item(self, idx):
        try:
            token_sequence = self._tokenize_npz(idx)
            token_sequence = self._truncate(token_sequence)
        except Exception:
            token_sequence = [BOS_TOKEN, TIME_SIG_OFFSET, BPM_OFFSET, BAR_TOKEN, EMPTY_MARKER, EOS_TOKEN]

        return {
            'skeleton_ids': torch.tensor(token_sequence, dtype=torch.long),
            'attention_mask': torch.ones(len(token_sequence), dtype=torch.long),
            'mask_positions': torch.zeros(0, dtype=torch.long),
            'mask_targets': torch.zeros(0, dtype=torch.long),
            'length': len(token_sequence),
            'num_masks': 0,
        }


class FELIXTaggerCollator:
    """Dynamic padding collator for FELIXTaggerDataset."""

    def __init__(self, pad_token_id=PAD_TOKEN, label_pad_id=LABEL_PAD, max_length=2048):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            seq_len = item['length']
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                input_ids_list.append(torch.cat([
                    item['input_ids'][:max_seq_len],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ]))
                attention_mask_list.append(torch.cat([
                    torch.ones(min(seq_len, max_seq_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
                labels_list.append(torch.cat([
                    item['labels'][:max_seq_len],
                    torch.full((pad_len,), self.label_pad_id, dtype=torch.long),
                ]))
            else:
                input_ids_list.append(item['input_ids'][:max_seq_len])
                attention_mask_list.append(torch.ones(max_seq_len, dtype=torch.long))
                labels_list.append(item['labels'][:max_seq_len])

        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
        }


class FELIXInserterCollator:
    """Dynamic padding collator for FELIXInserterDataset."""

    def __init__(self, pad_token_id=PAD_TOKEN, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )
        max_num_masks = max(item['num_masks'] for item in batch)
        max_num_masks = max(max_num_masks, 1)

        skeleton_ids_list = []
        attention_mask_list = []
        mask_positions_list = []
        mask_targets_list = []

        for item in batch:
            seq_len = item['length']
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                skeleton_ids_list.append(torch.cat([
                    item['skeleton_ids'][:max_seq_len],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ]))
                attention_mask_list.append(torch.cat([
                    torch.ones(min(seq_len, max_seq_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                skeleton_ids_list.append(item['skeleton_ids'][:max_seq_len])
                attention_mask_list.append(torch.ones(max_seq_len, dtype=torch.long))

            n_masks = item['num_masks']
            mask_pad = max_num_masks - n_masks

            if n_masks > 0:
                pos = item['mask_positions'][:max_num_masks]
                tgt = item['mask_targets'][:max_num_masks]
                # Filter out positions >= max_seq_len (truncation safety)
                valid = pos < max_seq_len
                pos = pos[valid]
                tgt = tgt[valid]
                n_masks = len(pos)
                mask_pad = max_num_masks - n_masks
            else:
                pos = torch.zeros(0, dtype=torch.long)
                tgt = torch.zeros(0, dtype=torch.long)

            if mask_pad > 0:
                mask_positions_list.append(torch.cat([
                    pos, torch.full((mask_pad,), -1, dtype=torch.long),
                ]))
                mask_targets_list.append(torch.cat([
                    tgt, torch.full((mask_pad,), self.pad_token_id, dtype=torch.long),
                ]))
            else:
                mask_positions_list.append(pos[:max_num_masks])
                mask_targets_list.append(tgt[:max_num_masks])

        return {
            'skeleton_ids': torch.stack(skeleton_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'mask_positions': torch.stack(mask_positions_list),
            'mask_targets': torch.stack(mask_targets_list),
        }


class BucketBatchSampler(Sampler):
    """Length-aware batch sampler for variable-length sequences."""

    def __init__(self, lengths, batch_size=32, bucket_size=200, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            self.buckets.append(sorted_indices[i:i + bucket_size])

    def __iter__(self):
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))
        for bi in bucket_order:
            bucket = list(self.buckets[bi])
            if self.shuffle:
                np.random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:
                    yield batch

    def __len__(self):
        return sum(
            (len(b) + self.batch_size - 1) // self.batch_size
            for b in self.buckets
        )


def get_file_lists(data_dir=DATA_DIR, test_ratio=0.05, val_ratio=0.05, seed=42):
    """Split npz files into train/val/test sets."""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)
    test_size = int(len(all_files) * test_ratio)
    val_size = int(len(all_files) * val_ratio)
    train_files = [all_files[i] for i in indices[:-(test_size + val_size)]]
    val_files = [all_files[i] for i in indices[-(test_size + val_size):-test_size]]
    test_files = [all_files[i] for i in indices[-test_size:]]
    return train_files, val_files, test_files
