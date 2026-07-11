"""
GECToR Dataset and Collator (no_pair encoding)

Loads npz piano roll files, tokenizes to no_pair encoding (absolute position),
generates perturbations, and extracts edit labels for training.
"""

import os
import sys
import random
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Optional

# Import local config first (before adding music_bert to path)
from config import (
    TRACK0_START, TRACK1_START,
    BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TIME_SIG_OFFSET, BPM_OFFSET, LABEL_PAD, NUM_LABELS,
    LABEL_KEEP, DATA_DIR,
)
from perturbation import perturb_sequence
from label_extractor import extract_labels

# Import PianoRollTokenizer from music_bert_no_pair
MUSIC_BERT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'music_bert_no_pair'
)
_spec = importlib.util.spec_from_file_location(
    "music_bert_no_pair_tokenizer",
    os.path.join(MUSIC_BERT_DIR, "my_tokenizer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PianoRollTokenizer = _mod.PianoRollTokenizer


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


class GECToRDataset(Dataset):
    """
    GECToR training dataset (no_pair encoding, absolute positions).

    Each __getitem__ dynamically:
    1. Loads npz and encodes to no_pair token sequence
    2. Generates a random perturbation (source)
    3. Extracts labels (source -> target)

    Args:
        file_list: list of npz file paths (absolute or relative to data_dir)
        data_dir: base directory for npz files
        max_len: maximum sequence length (default 2048)
        include_clean: whether to include clean samples (Stage III)
        clean_ratio: proportion of clean samples (default 0.0)
        p_pitch: pitch shift perturbation probability
        p_rhythm: rhythm change perturbation probability
        p_delete: note deletion perturbation probability
        p_insert: note insertion perturbation probability
        pitch_shift_augment: apply random pitch shift augmentation
    """

    def __init__(
        self,
        file_list: List[str],
        data_dir: str = DATA_DIR,
        max_len: int = 2048,
        include_clean: bool = False,
        clean_ratio: float = 0.0,
        p_pitch: float = 0.10,
        p_rhythm: float = 0.05,
        p_delete: float = 0.03,
        p_insert: float = 0.02,
        pitch_shift_augment: bool = False,
    ):
        self.file_list = file_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.include_clean = include_clean
        self.clean_ratio = clean_ratio
        self.p_pitch = p_pitch
        self.p_rhythm = p_rhythm
        self.p_delete = p_delete
        self.p_insert = p_insert
        self.pitch_shift_augment = pitch_shift_augment

        # Create tokenizer (matching music_bert_no_pair config)
        self.tokenizer = PianoRollTokenizer(
            patch_h=1,
            patch_w=4,
            pattern_num=81,
            beats_length=88,
        )

    def __len__(self):
        return len(self.file_list)

    def _tokenize_npz(self, idx):
        """
        Encode an npz file to no_pair token sequence (absolute position encoding).
        Replicates the logic from music_bert_no_pair/mlm_dataset.py.

        Key difference from Scheme B:
        - Uses compress_tokens with track_marker_id (TRACK0_START/TRACK1_START)
        - Absolute positions: [TRACK_MARKER][abs_pos][val][abs_pos][val]...
        - Empty beat: [TRACK_MARKER][0]
        """
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

        # BOS determination
        add_bos = True
        if '_' in file_name:
            suffix = os.path.basename(file_name).split('_')[-1].replace('.npz', '')
            if suffix.isdigit() and suffix != '1':
                add_bos = False

        # Optional pitch shift augmentation
        # NOTE: for GECToR, if we shift, both source and target get the same shift
        # (it's data augmentation, not an error)
        pitch_shift = 0
        if self.pitch_shift_augment and random.random() < 0.7:
            pitch_shift = random.randint(-5, 5)

        all_tokens = []

        # Header: BOS + TIME_SIG + BPM
        if add_bos:
            all_tokens.append(BOS_TOKEN)
        all_tokens.append(TIME_SIG_OFFSET + time_sig_idx)
        all_tokens.append(BPM_OFFSET + encode_bpm(bpm_value))

        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']
            measure = measure[:, ::-1, :].copy()  # reverse pitch axis

            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            t = measure.shape[2]
            beat_length = 4  # patch_w
            num_beats = (t + beat_length - 1) // beat_length

            # BAR token
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

                # Track 0 (high voice) - TRACK0_START marker + absolute positions
                p0 = beat_measure[:2]
                tok0 = self.tokenizer.image_to_patch_tokens(p0, strict_mode=True)
                comp0 = self.tokenizer.compress_tokens(
                    tok0,
                    track_marker_id=TRACK0_START,
                )
                all_tokens.extend(comp0.tolist())

                # Track 1 (low voice) - TRACK1_START marker + absolute positions
                p1 = beat_measure[2:]
                tok1 = self.tokenizer.image_to_patch_tokens(p1, strict_mode=True)
                comp1 = self.tokenizer.compress_tokens(
                    tok1,
                    track_marker_id=TRACK1_START,
                )
                all_tokens.extend(comp1.tolist())

        # EOS
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

    def __getitem__(self, idx):
        # 1. Load and tokenize
        token_sequence = self._tokenize_npz(idx)

        # 2. Truncate
        token_sequence = self._truncate(token_sequence)

        # 3. Generate perturbation or clean sample
        if self.include_clean and random.random() < self.clean_ratio:
            # Clean sample: source = target, all KEEP
            source_tokens = list(token_sequence)
            labels = [LABEL_KEEP] * len(source_tokens)
        else:
            # Perturbed sample
            source_tokens, target_tokens = perturb_sequence(
                token_sequence,
                p_pitch=self.p_pitch,
                p_rhythm=self.p_rhythm,
                p_delete=self.p_delete,
                p_insert=self.p_insert,
            )
            try:
                labels = extract_labels(source_tokens, target_tokens)
            except (AssertionError, Exception):
                # Fallback: treat as clean sample if label extraction fails
                source_tokens = list(token_sequence)
                labels = [LABEL_KEEP] * len(source_tokens)

        # 4. Generate binary error detection labels
        detect_labels = [0 if l == LABEL_KEEP else 1 for l in labels]

        return {
            'input_ids': torch.tensor(source_tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'detect_labels': torch.tensor(detect_labels, dtype=torch.long),
            'length': len(source_tokens),
        }


class GECToRCollator:
    """
    Dynamic padding collator for GECToR batches.
    """

    def __init__(self, pad_token_id=PAD_TOKEN, label_pad_id=LABEL_PAD, max_length=2048):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )

        input_ids = []
        labels = []
        detect_labels = []
        attention_mask = []

        for item in batch:
            L = item['length']
            pad_len = max_len - L

            if pad_len > 0:
                input_ids.append(torch.cat([
                    item['input_ids'][:max_len],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ]))
                labels.append(torch.cat([
                    item['labels'][:max_len],
                    torch.full((pad_len,), self.label_pad_id, dtype=torch.long),
                ]))
                detect_labels.append(torch.cat([
                    item['detect_labels'][:max_len],
                    torch.full((pad_len,), self.label_pad_id, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    torch.ones(min(L, max_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                input_ids.append(item['input_ids'][:max_len])
                labels.append(item['labels'][:max_len])
                detect_labels.append(item['detect_labels'][:max_len])
                attention_mask.append(torch.ones(max_len, dtype=torch.long))

        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'detect_labels': torch.stack(detect_labels),
            'attention_mask': torch.stack(attention_mask),
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
    """
    Split npz files into train/val/test sets.

    Returns:
        (train_files, val_files, test_files) as lists of filenames
    """
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)

    test_size = int(len(all_files) * test_ratio)
    val_size = int(len(all_files) * val_ratio)
    train_size = len(all_files) - test_size - val_size

    train_files = [all_files[i] for i in indices[:train_size]]
    val_files = [all_files[i] for i in indices[train_size:train_size + val_size]]
    test_files = [all_files[i] for i in indices[train_size + val_size:]]

    return train_files, val_files, test_files
