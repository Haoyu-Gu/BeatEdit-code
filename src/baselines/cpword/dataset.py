"""
CPWord GECToR Dataset and Collator

Loads MIDI files, tokenizes with miditok CPWord, generates perturbations,
and extracts factored edit labels for GECToR training.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict

from config import (
    PAD_ID, LABEL_PAD, ACTION_KEEP,
    BOS_TOKEN, EOS_TOKEN, MASK_ID, MASK_COMPOUND,
    NUM_SUBVOCABS, SUB_VOCAB_SIZES,
    FAMILY_METRIC, FAMILY_NOTE,
    MIDI_DATA_DIR,
    is_music_token,
)
from cp_tokenizer import midi_to_compound_tokens
from perturbation import perturb_sequence
from label_extractor import extract_labels


class CPGECToRDataset(Dataset):
    """
    GECToR training dataset using CPWord encoding.

    Each __getitem__ dynamically:
    1. Loads MIDI and tokenizes to CPWord
    2. Adds BOS/EOS compound tokens
    3. Generates a random perturbation (source)
    4. Extracts factored labels (source -> target)
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
        """Tokenize a MIDI file to CPWord compound token sequence with BOS/EOS."""
        file_name = self.file_list[idx]
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            file_path = os.path.join(self.data_dir, file_name)

        tokens = midi_to_compound_tokens(file_path)

        # Add BOS at start if not present
        if len(tokens) == 0 or tokens[0] != BOS_TOKEN:
            tokens = [list(BOS_TOKEN)] + tokens

        # Add EOS at end if not present
        if tokens[-1] != EOS_TOKEN:
            tokens.append(list(EOS_TOKEN))

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
                source_tokens = [list(t) for t in token_sequence]
                action_labels = [ACTION_KEEP] * len(source_tokens)
                sub_token_labels = [[-100] * 5 for _ in range(len(source_tokens))]
            else:
                source_tokens, target_tokens = perturb_sequence(
                    token_sequence,
                    p_pitch=self.p_pitch,
                    p_rhythm=self.p_rhythm,
                    p_delete=self.p_delete,
                    p_insert=self.p_insert,
                )
                try:
                    action_labels, sub_token_labels = extract_labels(
                        source_tokens, target_tokens)
                except Exception:
                    source_tokens = [list(t) for t in token_sequence]
                    action_labels = [ACTION_KEEP] * len(source_tokens)
                    sub_token_labels = [[-100] * 5 for _ in range(len(source_tokens))]

            # Re-truncate if perturbation changed length
            if len(source_tokens) > self.max_len:
                source_tokens = source_tokens[:self.max_len]
                action_labels = action_labels[:self.max_len]
                sub_token_labels = sub_token_labels[:self.max_len]

            # 4. Generate binary error detection labels
            detect_labels = [0 if a == ACTION_KEEP else 1 for a in action_labels]

            return {
                'compound_ids': torch.tensor(source_tokens, dtype=torch.long),  # (N, 5)
                'action_labels': torch.tensor(action_labels, dtype=torch.long),  # (N,)
                'sub_token_labels': torch.tensor(sub_token_labels, dtype=torch.long),  # (N, 5)
                'detect_labels': torch.tensor(detect_labels, dtype=torch.long),  # (N,)
                'length': len(source_tokens),
            }
        except Exception:
            # Fallback: return a minimal valid sample
            dummy = [list(BOS_TOKEN), list(EOS_TOKEN)]
            return {
                'compound_ids': torch.tensor(dummy, dtype=torch.long),
                'action_labels': torch.tensor([ACTION_KEEP, ACTION_KEEP], dtype=torch.long),
                'sub_token_labels': torch.tensor([[-100]*5, [-100]*5], dtype=torch.long),
                'detect_labels': torch.tensor([0, 0], dtype=torch.long),
                'length': 2,
            }


class CPMLMDataset(Dataset):
    """
    MLM pretraining dataset using CPWord encoding.

    Masks 15% of music compound tokens: 80% -> MASK, 10% -> random, 10% -> keep.
    Masks all 5 sub-tokens of a compound token together.
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

        tokens = midi_to_compound_tokens(file_path)
        if len(tokens) == 0 or tokens[0] != BOS_TOKEN:
            tokens = [list(BOS_TOKEN)] + tokens
        if tokens[-1] != EOS_TOKEN:
            tokens.append(list(EOS_TOKEN))
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
        try:
            tokens = self._tokenize_midi(idx)
            tokens = self._truncate(tokens)

            input_ids = [list(t) for t in tokens]
            mlm_labels = [[-100] * 5 for _ in range(len(tokens))]

            for i in range(len(tokens)):
                if not is_music_token(tokens[i]):
                    continue

                if random.random() < self.mask_prob:
                    # Set label to original sub-tokens
                    mlm_labels[i] = list(tokens[i])

                    r = random.random()
                    if r < 0.8:
                        # MASK all sub-tokens
                        input_ids[i] = list(MASK_COMPOUND)
                    elif r < 0.9:
                        # Random compound token
                        input_ids[i] = _random_compound_token(tokens[i])
                    # else: keep original (10%)

            return {
                'compound_ids': torch.tensor(input_ids, dtype=torch.long),  # (N, 5)
                'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),  # (N, 5)
                'length': len(input_ids),
            }
        except Exception:
            dummy = [list(BOS_TOKEN), list(EOS_TOKEN)]
            return {
                'compound_ids': torch.tensor(dummy, dtype=torch.long),
                'mlm_labels': torch.tensor([[-100]*5, [-100]*5], dtype=torch.long),
                'length': 2,
            }


def _random_compound_token(original):
    """Generate a random compound token of the same type."""
    from config import (
        FAMILY_METRIC, FAMILY_NOTE, POS_BAR, POS_OFFSET, POS_MAX,
        POS_IGNORE, PITCH_IGNORE, PITCH_MIN_ID, PITCH_MAX_ID,
        VEL_IGNORE, VEL_OFFSET, VEL_MAX_ID,
        DUR_IGNORE, DUR_OFFSET, DUR_MAX_ID,
    )

    if original[0] == FAMILY_METRIC:
        if original[1] == POS_BAR:
            # Bar token -> keep as is
            return list(original)
        else:
            # Position token -> random position
            return [FAMILY_METRIC, random.randint(POS_OFFSET, POS_MAX),
                    PITCH_IGNORE, VEL_IGNORE, DUR_IGNORE]
    elif original[0] == FAMILY_NOTE:
        # Note token -> random note
        return [FAMILY_NOTE, POS_IGNORE,
                random.randint(PITCH_MIN_ID, PITCH_MAX_ID),
                random.randint(VEL_OFFSET, VEL_MAX_ID),
                random.randint(DUR_OFFSET, DUR_MAX_ID)]
    return list(original)


class CPGECToRCollator:
    """Dynamic padding collator for CPWord GECToR batches."""

    def __init__(self, max_length=2048):
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )

        compound_ids = []
        action_labels = []
        sub_token_labels = []
        detect_labels = []
        attention_mask = []

        for item in batch:
            L = item['length']
            pad_len = max_len - L

            if pad_len > 0:
                # Compound IDs: pad with [0,0,0,0,0]
                compound_ids.append(torch.cat([
                    item['compound_ids'][:max_len],
                    torch.zeros((pad_len, NUM_SUBVOCABS), dtype=torch.long),
                ]))
                action_labels.append(torch.cat([
                    item['action_labels'][:max_len],
                    torch.full((pad_len,), LABEL_PAD, dtype=torch.long),
                ]))
                sub_token_labels.append(torch.cat([
                    item['sub_token_labels'][:max_len],
                    torch.full((pad_len, NUM_SUBVOCABS), LABEL_PAD, dtype=torch.long),
                ]))
                detect_labels.append(torch.cat([
                    item['detect_labels'][:max_len],
                    torch.full((pad_len,), LABEL_PAD, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    torch.ones(min(L, max_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                compound_ids.append(item['compound_ids'][:max_len])
                action_labels.append(item['action_labels'][:max_len])
                sub_token_labels.append(item['sub_token_labels'][:max_len])
                detect_labels.append(item['detect_labels'][:max_len])
                attention_mask.append(torch.ones(max_len, dtype=torch.long))

        return {
            'compound_ids': torch.stack(compound_ids),       # (B, L, 5)
            'action_labels': torch.stack(action_labels),     # (B, L)
            'sub_token_labels': torch.stack(sub_token_labels), # (B, L, 5)
            'detect_labels': torch.stack(detect_labels),     # (B, L)
            'attention_mask': torch.stack(attention_mask),    # (B, L)
        }


class CPMLMCollator:
    """Dynamic padding collator for CPWord MLM pretraining."""

    def __init__(self, max_length=2048):
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(item['length'] for item in batch),
            self.max_length,
        )

        compound_ids = []
        mlm_labels = []
        attention_mask = []

        for item in batch:
            L = item['length']
            pad_len = max_len - L

            if pad_len > 0:
                compound_ids.append(torch.cat([
                    item['compound_ids'][:max_len],
                    torch.zeros((pad_len, NUM_SUBVOCABS), dtype=torch.long),
                ]))
                mlm_labels.append(torch.cat([
                    item['mlm_labels'][:max_len],
                    torch.full((pad_len, NUM_SUBVOCABS), -100, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    torch.ones(min(L, max_len), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
            else:
                compound_ids.append(item['compound_ids'][:max_len])
                mlm_labels.append(item['mlm_labels'][:max_len])
                attention_mask.append(torch.ones(max_len, dtype=torch.long))

        return {
            'compound_ids': torch.stack(compound_ids),   # (B, L, 5)
            'mlm_labels': torch.stack(mlm_labels),       # (B, L, 5)
            'attention_mask': torch.stack(attention_mask), # (B, L)
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
    """Split MIDI files into train/val/test sets (same seed=42 split as BEAT)."""
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
