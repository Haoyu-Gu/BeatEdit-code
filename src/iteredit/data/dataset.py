"""
Dataset for Levenshtein Transformer Music Inpainting.

Provides:
- LevTDataset: online mask + intermediate state sampling + label generation
- LevTCollator: dynamic padding for variable-length sequences
- BucketBatchSampler: length-aware batching
- Data split utilities
"""

import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict

from configs.config import (
    EMPTY_MARKER, SPLIT_0, SPLIT_1, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    PLH_TOKEN, VOCAB_SIZE, TIME_SIG_OFFSET, BPM_OFFSET, DATA_DIR,
)
from data.tokenizer import create_tokenizer
from data.masking import create_inpainting_pair, sample_intermediate_state
from data.levenshtein_utils import compute_edit_labels_with_context


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


class LevTDataset(Dataset):
    """
    Dataset for Levenshtein Transformer music inpainting.

    Each sample:
    1. Tokenize NPZ to BEAT sequence
    2. Create inpainting pair (mask consecutive beats)
    3. Sample intermediate state (corrupt mask region)
    4. Compute Levenshtein edit labels (del, ins, tok)
    """

    def __init__(
        self,
        file_list: List[str],
        data_dir: str = DATA_DIR,
        max_len: int = 2048,
        mask_ratio_min: float = 0.125,
        mask_ratio_max: float = 0.5,
        corruption_delete_prob: float = 0.3,
        corruption_replace_prob: float = 0.2,
        pitch_shift_augment: bool = False,
        max_insert: int = 20,
    ):
        self.file_list = file_list
        self.data_dir = data_dir
        self.max_len = max_len
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.corruption_delete_prob = corruption_delete_prob
        self.corruption_replace_prob = corruption_replace_prob
        self.pitch_shift_augment = pitch_shift_augment
        self.max_insert = max_insert
        self.tokenizer = create_tokenizer()

    def __len__(self):
        return len(self.file_list)

    def _tokenize_npz(self, idx):
        """Encode an npz file to with_pair bundled token sequence."""
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

    def __getitem__(self, idx):
        try:
            return self._get_item_impl(idx)
        except Exception:
            return self._fallback_item()

    def _get_item_impl(self, idx):
        # 1. Tokenize
        full_tokens = self._tokenize_npz(idx)
        full_tokens = self._truncate(full_tokens)

        # 2. Create inpainting pair (mask consecutive beats)
        pair = create_inpainting_pair(
            full_tokens,
            mask_ratio_min=self.mask_ratio_min,
            mask_ratio_max=self.mask_ratio_max,
        )
        if pair is None:
            return self._fallback_item()

        target_ids = pair['full_ids']
        mask_start = pair['mask_start']
        mask_end = pair['mask_end']

        # 3. Sample intermediate state (corrupt the mask region of target)
        intermediate, new_mask_start, new_mask_end = sample_intermediate_state(
            target_ids, mask_start, mask_end,
            vocab_size=VOCAB_SIZE,
            delete_prob=self.corruption_delete_prob,
            replace_prob=self.corruption_replace_prob,
        )

        # 4. Compute Levenshtein edit labels
        del_labels, ins_labels, tok_labels = compute_edit_labels_with_context(
            intermediate, target_ids, new_mask_start, new_mask_end,
        )

        # Clamp ins_labels to max_insert
        ins_labels = [min(x, self.max_insert) for x in ins_labels]

        # 5. Build context mask (1 = editable/mask region, 0 = context/frozen)
        context_mask = [0] * len(intermediate)
        for i in range(new_mask_start, new_mask_end):
            if i < len(context_mask):
                context_mask[i] = 1

        # 6. Build tok_input_ids: intermediate with PLH tokens inserted at gaps
        #    indicated by ins_labels.  Also record PLH positions and their targets.
        tok_input_ids = []
        tok_positions = []
        tok_target_idx = 0  # pointer into tok_labels
        for i in range(len(intermediate) + 1):
            n_ins = ins_labels[i]
            for k in range(n_ins):
                tok_positions.append(len(tok_input_ids))
                tok_input_ids.append(PLH_TOKEN)
                tok_target_idx += 1
            if i < len(intermediate):
                tok_input_ids.append(intermediate[i])

        # tok_targets mirrors tok_labels (one per PLH position)
        tok_targets = list(tok_labels) if tok_labels else []

        return {
            'z_ids': torch.tensor(intermediate, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'del_labels': torch.tensor(del_labels, dtype=torch.long),
            'ins_labels': torch.tensor(ins_labels, dtype=torch.long),
            'tok_labels': torch.tensor(tok_labels, dtype=torch.long) if tok_labels else torch.zeros(0, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.float),
            'z_length': len(intermediate),
            'target_length': len(target_ids),
            'num_tok_labels': len(tok_labels),
            'tok_input_ids': torch.tensor(tok_input_ids, dtype=torch.long),
            'tok_positions': torch.tensor(tok_positions, dtype=torch.long) if tok_positions else torch.zeros(0, dtype=torch.long),
            'tok_targets': torch.tensor(tok_targets, dtype=torch.long) if tok_targets else torch.zeros(0, dtype=torch.long),
            'tok_input_length': len(tok_input_ids),
            'num_tok_positions': len(tok_positions),
        }

    def _fallback_item(self):
        """Minimal valid sample for error cases."""
        tokens = [BOS_TOKEN, TIME_SIG_OFFSET, BPM_OFFSET, BAR_TOKEN, EMPTY_MARKER, EOS_TOKEN]
        z_len = len(tokens)
        return {
            'z_ids': torch.tensor(tokens, dtype=torch.long),
            'target_ids': torch.tensor(tokens, dtype=torch.long),
            'del_labels': torch.zeros(z_len, dtype=torch.long),
            'ins_labels': torch.zeros(z_len + 1, dtype=torch.long),
            'tok_labels': torch.zeros(0, dtype=torch.long),
            'context_mask': torch.zeros(z_len, dtype=torch.float),
            'z_length': z_len,
            'target_length': z_len,
            'num_tok_labels': 0,
            'tok_input_ids': torch.tensor(tokens, dtype=torch.long),
            'tok_positions': torch.zeros(0, dtype=torch.long),
            'tok_targets': torch.zeros(0, dtype=torch.long),
            'tok_input_length': z_len,
            'num_tok_positions': 0,
        }


class LevTCollator:
    """
    Dynamic padding collator for LevTDataset.

    Pads z_ids, del_labels, context_mask to max z_length in batch.
    Pads ins_labels to max (z_length + 1) in batch.
    Pads tok_labels to max num_tok_labels in batch.
    Pads target_ids to max target_length in batch.
    """

    def __init__(self, pad_token_id=PAD_TOKEN, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_z_len = min(max(item['z_length'] for item in batch), self.max_length)
        max_tgt_len = min(max(item['target_length'] for item in batch), self.max_length)
        max_ins_len = max_z_len + 1
        max_tok_len = max(max(item['num_tok_labels'] for item in batch), 1)

        # New fields for token loss
        max_tok_input_len = min(max(item['tok_input_length'] for item in batch), self.max_length)
        max_tok_pos_len = max(max(item['num_tok_positions'] for item in batch), 1)

        z_ids_list = []
        attention_mask_list = []
        del_labels_list = []
        ins_labels_list = []
        tok_labels_list = []
        context_mask_list = []
        target_ids_list = []
        tok_input_ids_list = []
        tok_attention_mask_list = []
        tok_positions_list = []
        tok_targets_list = []

        for item in batch:
            z_len = min(item['z_length'], max_z_len)
            tgt_len = min(item['target_length'], max_tgt_len)

            # z_ids
            z = item['z_ids'][:z_len]
            z_pad = max_z_len - z_len
            if z_pad > 0:
                z = torch.cat([z, torch.full((z_pad,), self.pad_token_id, dtype=torch.long)])
            z_ids_list.append(z)

            # attention_mask
            attn = torch.cat([
                torch.ones(z_len, dtype=torch.long),
                torch.zeros(z_pad, dtype=torch.long),
            ]) if z_pad > 0 else torch.ones(z_len, dtype=torch.long)
            attention_mask_list.append(attn)

            # del_labels
            dl = item['del_labels'][:z_len]
            if z_pad > 0:
                dl = torch.cat([dl, torch.full((z_pad,), -100, dtype=torch.long)])
            del_labels_list.append(dl)

            # ins_labels (z_len + 1 gaps)
            il_len = min(item['ins_labels'].size(0), z_len + 1)
            il = item['ins_labels'][:il_len]
            il_pad = max_ins_len - il_len
            if il_pad > 0:
                il = torch.cat([il, torch.full((il_pad,), 0, dtype=torch.long)])
            ins_labels_list.append(il[:max_ins_len])

            # tok_labels
            tl = item['tok_labels']
            tl_len = min(tl.size(0), max_tok_len)
            tl = tl[:tl_len]
            tl_pad = max_tok_len - tl_len
            if tl_pad > 0:
                tl = torch.cat([tl, torch.full((tl_pad,), self.pad_token_id, dtype=torch.long)])
            tok_labels_list.append(tl)

            # context_mask
            cm = item['context_mask'][:z_len]
            if z_pad > 0:
                cm = torch.cat([cm, torch.zeros(z_pad, dtype=torch.float)])
            context_mask_list.append(cm)

            # target_ids
            ti = item['target_ids'][:tgt_len]
            ti_pad = max_tgt_len - tgt_len
            if ti_pad > 0:
                ti = torch.cat([ti, torch.full((ti_pad,), self.pad_token_id, dtype=torch.long)])
            target_ids_list.append(ti)

            # --- tok_input_ids (intermediate with PLH inserted) ---
            tok_in_len = min(item['tok_input_length'], max_tok_input_len)
            tok_in = item['tok_input_ids'][:tok_in_len]
            tok_in_pad = max_tok_input_len - tok_in_len
            if tok_in_pad > 0:
                tok_in = torch.cat([tok_in, torch.full((tok_in_pad,), self.pad_token_id, dtype=torch.long)])
            tok_input_ids_list.append(tok_in)

            # tok_attention_mask
            tok_attn = torch.cat([
                torch.ones(tok_in_len, dtype=torch.long),
                torch.zeros(tok_in_pad, dtype=torch.long),
            ]) if tok_in_pad > 0 else torch.ones(tok_in_len, dtype=torch.long)
            tok_attention_mask_list.append(tok_attn)

            # --- tok_positions (indices where PLH tokens are in tok_input_ids) ---
            tp = item['tok_positions']
            tp_len = min(tp.size(0), max_tok_pos_len)
            # Filter out positions that exceed max_tok_input_len
            if tp_len > 0:
                tp = tp[:tp_len]
                tp = tp[tp < max_tok_input_len]  # keep only valid positions
                tp_len = tp.size(0)
            tp_pad = max_tok_pos_len - tp_len
            if tp_pad > 0:
                tp = torch.cat([tp, torch.full((tp_pad,), -1, dtype=torch.long)])
            tok_positions_list.append(tp[:max_tok_pos_len])

            # --- tok_targets (target token for each PLH position) ---
            tt = item['tok_targets']
            tt_len = min(tt.size(0), max_tok_pos_len)
            tt = tt[:tt_len]
            tt_pad = max_tok_pos_len - tt_len
            if tt_pad > 0:
                tt = torch.cat([tt, torch.full((tt_pad,), self.pad_token_id, dtype=torch.long)])
            tok_targets_list.append(tt[:max_tok_pos_len])

        return {
            'z_ids': torch.stack(z_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'del_labels': torch.stack(del_labels_list),
            'ins_labels': torch.stack(ins_labels_list),
            'tok_labels': torch.stack(tok_labels_list),
            'context_mask': torch.stack(context_mask_list),
            'target_ids': torch.stack(target_ids_list),
            'tok_input_ids': torch.stack(tok_input_ids_list),
            'tok_attention_mask': torch.stack(tok_attention_mask_list),
            'tok_positions': torch.stack(tok_positions_list),
            'tok_targets': torch.stack(tok_targets_list),
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


def load_lengths_cache(data_dir, file_list):
    """Try to load cached lengths or estimate."""
    cache_path = os.path.join(data_dir, '.lengths_cache_with_pair.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_lengths = pickle.load(f)
        lengths = []
        for fname in file_list:
            lengths.append(all_lengths.get(fname, 1000))
        return lengths
    return [1000] * len(file_list)
