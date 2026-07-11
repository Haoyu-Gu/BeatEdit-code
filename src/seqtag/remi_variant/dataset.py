"""
REMI GECToR Dataset and Collator

Loads MIDI files, tokenizes with miditok REMI, generates perturbations,
and extracts edit labels for GECToR training.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict

from config import (
    PAD_TOKEN, LABEL_PAD, LABEL_KEEP,
    BOS_TOKEN, EOS_TOKEN, MIDI_DATA_DIR,
)
from remi_tokenizer import midi_to_tokens
from perturbation import perturb_sequence
from label_extractor import extract_labels


class REMIGECToRDataset(Dataset):
    """
    GECToR training dataset using REMI encoding.

    Each __getitem__ dynamically:
    1. Loads MIDI and tokenizes to REMI
    2. Adds BOS/EOS tokens
    3. Generates a random perturbation (source)
    4. Extracts labels (source -> target)
    """

    def __init__(
        self,
        file_list: List[str],
        data_dir: str = MIDI_DATA_DIR,
        max_len: int = 2048,
        include_clean: bool = False,
        clean_ratio: float = 0.0,
        p_pitch: float = 0.10,
        p_rhythm: float = 0.05,
        p_delete: float = 0.03,
        p_insert: float = 0.02,
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

    def __len__(self):
        return len(self.file_list)

    def _tokenize_midi(self, idx):
        """Tokenize a MIDI file to REMI token sequence with BOS/EOS."""
        file_name = self.file_list[idx]
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            file_path = os.path.join(self.data_dir, file_name)

        tokens = midi_to_tokens(file_path)

        # Add BOS at start if not present
        if len(tokens) == 0 or tokens[0] != BOS_TOKEN:
            tokens = [BOS_TOKEN] + tokens

        # Add EOS at end if not present
        if tokens[-1] != EOS_TOKEN:
            tokens.append(EOS_TOKEN)

        return tokens

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
        try:
            # 1. Load and tokenize
            token_sequence = self._tokenize_midi(idx)

            # 2. Truncate
            token_sequence = self._truncate(token_sequence)

            # 3. Generate perturbation or clean sample
            if self.include_clean and random.random() < self.clean_ratio:
                source_tokens = list(token_sequence)
                labels = [LABEL_KEEP] * len(source_tokens)
            else:
                source_tokens, target_tokens = perturb_sequence(
                    token_sequence,
                    p_pitch=self.p_pitch,
                    p_rhythm=self.p_rhythm,
                    p_delete=self.p_delete,
                    p_insert=self.p_insert,
                )
                try:
                    labels = extract_labels(source_tokens, target_tokens)
                except Exception:
                    source_tokens = list(token_sequence)
                    labels = [LABEL_KEEP] * len(source_tokens)

            # Re-truncate if perturbation changed length
            if len(source_tokens) > self.max_len:
                source_tokens = source_tokens[:self.max_len]
                labels = labels[:self.max_len]

            # 4. Generate binary error detection labels
            detect_labels = [0 if l == LABEL_KEEP else 1 for l in labels]

            return {
                'input_ids': torch.tensor(source_tokens, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'detect_labels': torch.tensor(detect_labels, dtype=torch.long),
                'length': len(source_tokens),
            }
        except Exception:
            # Fallback: return a minimal valid sample
            dummy = [BOS_TOKEN, EOS_TOKEN]
            return {
                'input_ids': torch.tensor(dummy, dtype=torch.long),
                'labels': torch.tensor([LABEL_KEEP, LABEL_KEEP], dtype=torch.long),
                'detect_labels': torch.tensor([0, 0], dtype=torch.long),
                'length': 2,
            }


class REMIMLMDataset(Dataset):
    """
    MLM pretraining dataset using REMI encoding.

    Masks 15% of music tokens: 80% -> MASK, 10% -> random, 10% -> keep.
    """

    def __init__(
        self,
        file_list: List[str],
        data_dir: str = MIDI_DATA_DIR,
        max_len: int = 2048,
        mask_prob: float = 0.15,
    ):
        self.file_list = file_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.file_list)

    def _tokenize_midi(self, idx):
        file_name = self.file_list[idx]
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            file_path = os.path.join(self.data_dir, file_name)

        tokens = midi_to_tokens(file_path)
        if len(tokens) == 0 or tokens[0] != BOS_TOKEN:
            tokens = [BOS_TOKEN] + tokens
        if tokens[-1] != EOS_TOKEN:
            tokens.append(EOS_TOKEN)
        return tokens

    def _truncate(self, tokens):
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
        from config import (
            MASK_TOKEN, MUSIC_TOKEN_MIN, MUSIC_TOKEN_MAX,
            is_maskable_token, VOCAB_SIZE,
        )
        try:
            tokens = self._tokenize_midi(idx)
            tokens = self._truncate(tokens)

            input_ids = list(tokens)
            mlm_labels = [-100] * len(tokens)

            for i in range(len(tokens)):
                if not is_maskable_token(tokens[i]):
                    continue

                if random.random() < self.mask_prob:
                    mlm_labels[i] = tokens[i]  # original token as label

                    r = random.random()
                    if r < 0.8:
                        input_ids[i] = MASK_TOKEN
                    elif r < 0.9:
                        input_ids[i] = random.randint(MUSIC_TOKEN_MIN, MUSIC_TOKEN_MAX)
                    # else: keep original (10%)

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(mlm_labels, dtype=torch.long),
                'length': len(input_ids),
            }
        except Exception:
            dummy = [BOS_TOKEN, EOS_TOKEN]
            return {
                'input_ids': torch.tensor(dummy, dtype=torch.long),
                'labels': torch.tensor([-100, -100], dtype=torch.long),
                'length': 2,
            }


class REMICollator:
    """Dynamic padding collator for REMI batches (GECToR)."""

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


class MLMCollator:
    """Dynamic padding collator for MLM pretraining."""

    def __init__(self, pad_token_id=PAD_TOKEN, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )

        input_ids = []
        labels = []
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
                    torch.full((pad_len,), -100, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    torch.ones(min(L, max_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                input_ids.append(item['input_ids'][:max_len])
                labels.append(item['labels'][:max_len])
                attention_mask.append(torch.ones(max_len, dtype=torch.long))

        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
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


def get_file_lists(data_dir=MIDI_DATA_DIR, test_ratio=0.05, val_ratio=0.05, seed=42):
    """
    Split MIDI files into train/val/test sets.

    Uses the same seed=42 split as BEAT encoding for fair comparison.
    """
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mid')])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)

    test_size = int(len(all_files) * test_ratio)
    val_size = int(len(all_files) * val_ratio)

    train_files = [all_files[i] for i in indices[:-(test_size + val_size)]]
    val_files = [all_files[i] for i in indices[-(test_size + val_size):-test_size]]
    test_files = [all_files[i] for i in indices[-test_size:]]

    return train_files, val_files, test_files
