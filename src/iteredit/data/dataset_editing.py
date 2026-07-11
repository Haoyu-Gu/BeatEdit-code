"""
Dataset for Levenshtein Transformer Editing Training.

Uses SchemeLoader (from evaluation) for scheme-specific tokenization,
parsing, perturbation, and sequence rebuilding.

Flow per sample:
1. Tokenize NPZ → target_tokens
2. Parse → separate melody/accomp → perturb accomp
3. Build intermediate state (corrupt changed accomp beats of target)
4. Compute per-beat Levenshtein labels (intermediate → target)
5. Return tensors for LevT three-head training

Usage:
    from data.dataset_editing import LevTEditingDataset, LevTEditingCollator
"""

import os
import sys
import copy
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# LevT project root
LEVT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIOUS_DIR = os.path.dirname(LEVT_DIR)
UNIFIED_DIR = os.path.join(PREVIOUS_DIR, 'evaluation')
sys.path.insert(0, LEVT_DIR)
sys.path.insert(0, UNIFIED_DIR)

from scheme_utils import SchemeLoader, SCHEME_INFO
from data.levenshtein_utils import compute_edit_labels

DATA_DIR = os.environ.get("BEATEDIT_DATA_DIR", "/path/to/data/npz")

# ==================== Scheme Token Constants ====================
# Keyed by scheme letter. Each entry has the special tokens needed.
SCHEME_TOKENS = {
    'A': {
        'vocab_size': 186, 'pad': 173, 'bos': 172, 'eos': 171,
        'plh': 185, 'bar': 170, 'note_max': 168,
        'track0': 183, 'track1': 184, 'control_min': 170,
    },
    'B': {
        'vocab_size': 185, 'pad': 174, 'bos': 172, 'eos': 171,
        'plh': 184, 'bar': 170, 'note_max': 168,
        'track0': None, 'track1': None, 'control_min': 170,
    },
    'C': {
        'vocab_size': 7146, 'pad': 7134, 'bos': 7133, 'eos': 7132,
        'plh': 7145, 'bar': 7131, 'note_max': 7127,
        'track0': 7129, 'track1': 7130, 'control_min': 7128,
    },
    'D': {
        'vocab_size': 7146, 'pad': 7134, 'bos': 7133, 'eos': 7132,
        'plh': 7145, 'bar': 7131, 'note_max': 7127,
        'track0': 7129, 'track1': 7130, 'control_min': 7128,
    },
}


def encode_bpm(bpm):
    if bpm is None:
        return 3
    bpm = int(bpm)
    if bpm < 90:
        return 0
    elif bpm <= 200:
        return 1
    else:
        return 2


class LevTEditingDataset(Dataset):
    """
    Dataset for LevT editing training via perturbation.

    Unlike the inpainting dataset (which masks contiguous beats),
    this dataset perturbs scattered accompaniment beats and computes
    per-beat Levenshtein labels between the corrupted intermediate
    state and the original target.
    """

    def __init__(
        self,
        file_list,
        scheme='A',
        data_dir=DATA_DIR,
        max_len=2048,
        corruption_delete_prob=0.3,
        corruption_replace_prob=0.2,
        pitch_shift_augment=False,
        max_insert=20,
    ):
        self.file_list = file_list
        self.scheme = scheme
        self.data_dir = data_dir
        self.max_len = max_len
        self.corruption_delete_prob = corruption_delete_prob
        self.corruption_replace_prob = corruption_replace_prob
        self.pitch_shift_augment = pitch_shift_augment
        self.max_insert = max_insert

        self.tok = SCHEME_TOKENS[scheme]

        # Initialize SchemeLoader (loads FELIX modules once)
        self._loader = SchemeLoader(scheme)

        # Create a reusable FELIX dataset for fast tokenization
        felix = self._loader.felix
        self._felix_ds = felix._FELIXBaseDataset(
            file_list=file_list,
            data_dir=data_dir,
            max_len=4096,  # tokenize full, truncate later
            pitch_shift_augment=pitch_shift_augment,
        )
        self._parse_fn = felix._parse_sequence
        self._separate_fn = felix._separate_tracks
        self._rebuild_fn = felix._rebuild_interleaved
        self._perturb_fn = felix._perturb_accompaniment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            return self._get_item_impl(idx)
        except Exception:
            return self._fallback_item()

    def _truncate(self, tokens):
        if len(tokens) <= self.max_len:
            return tokens
        p = random.random()
        if p < 0.33:
            return tokens[:self.max_len]
        elif p < 0.66:
            return tokens[-self.max_len:]
        else:
            start = random.randint(0, len(tokens) - self.max_len)
            return tokens[start:start + self.max_len]

    def _corrupt_tokens(self, beat_tokens):
        """Corrupt beat tokens to simulate an intermediate refinement state."""
        note_max = self.tok['note_max']
        corrupted = []
        for tok in beat_tokens:
            r = random.random()
            if r < self.corruption_delete_prob:
                continue  # delete
            elif r < self.corruption_delete_prob + self.corruption_replace_prob:
                corrupted.append(random.randint(0, note_max))
            else:
                corrupted.append(tok)  # keep correct
        return corrupted

    def _get_item_impl(self, idx):
        PAD = self.tok['pad']
        PLH = self.tok['plh']

        # 1. Tokenize
        target_tokens = self._felix_ds._tokenize_npz(idx)
        target_tokens = self._truncate(target_tokens)

        # 2. Parse and separate tracks
        parsed = self._parse_fn(target_tokens)
        mel_beats, acc_beats = self._separate_fn(parsed)

        if len(acc_beats) < 2:
            return self._fallback_item()

        # 3. Perturb accompaniment → get changed_mask
        _, level, changed_mask = self._perturb_fn(copy.deepcopy(acc_beats))
        changed_indices = [i for i, m in enumerate(changed_mask) if m]

        if len(changed_indices) == 0:
            return self._fallback_item()

        # 4. Create intermediate: target with changed accomp beats corrupted
        inter_acc = copy.deepcopy(acc_beats)
        for i in changed_indices:
            inter_acc[i] = copy.deepcopy(acc_beats[i])
            inter_acc[i]['tokens'] = self._corrupt_tokens(acc_beats[i]['tokens'])

        intermediate_tokens = self._rebuild_fn(mel_beats, inter_acc, parsed)

        if len(intermediate_tokens) > self.max_len:
            intermediate_tokens = intermediate_tokens[:self.max_len]

        # 5. Re-parse intermediate and target to get beat positions
        inter_parsed = self._parse_fn(intermediate_tokens)
        _, inter_acc_beats = self._separate_fn(inter_parsed)

        tgt_parsed = self._parse_fn(target_tokens)
        _, tgt_acc_beats = self._separate_fn(tgt_parsed)

        # 6. Compute per-beat Levenshtein labels
        z_len = len(intermediate_tokens)
        del_labels = [-100] * z_len       # -100 = ignore (context)
        ins_labels = [0] * (z_len + 1)    # 0 = no insertion
        tok_labels = []
        context_mask = [0.0] * z_len      # 0 = frozen context

        n_beats = min(len(changed_mask), len(inter_acc_beats), len(tgt_acc_beats))

        for i in range(n_beats):
            if not changed_mask[i]:
                continue

            inter_beat = inter_acc_beats[i]
            tgt_beat = tgt_acc_beats[i]

            z_beat = inter_beat['tokens']   # beat content without track marker
            y_beat = tgt_beat['tokens']

            # Skip if beat is out of truncation range
            if inter_beat['start_idx'] >= z_len:
                continue

            beat_del, beat_ins, beat_tok = compute_edit_labels(z_beat, y_beat)

            # Note tokens start after the track marker
            # start_idx points to track marker, note content is [start_idx+1, end_idx)
            note_start = inter_beat['start_idx'] + 1

            # Map del_labels
            for j, d in enumerate(beat_del):
                pos = note_start + j
                if pos < z_len:
                    del_labels[pos] = d
                    context_mask[pos] = 1.0

            # Map ins_labels
            # beat_ins[j] = insertions before z_beat[j]
            # beat_ins[len(z_beat)] = insertions after last token
            for j, n in enumerate(beat_ins):
                pos = note_start + j
                if pos <= z_len:
                    ins_labels[pos] = n

            tok_labels.extend(beat_tok)

        # 7. Clamp ins_labels
        ins_labels = [min(x, self.max_insert) for x in ins_labels]

        # 8. Build tok_input_ids (intermediate with PLH tokens inserted at gaps)
        tok_input_ids = []
        tok_positions = []
        for i in range(z_len + 1):
            n_ins = ins_labels[i]
            for _ in range(n_ins):
                tok_positions.append(len(tok_input_ids))
                tok_input_ids.append(PLH)
            if i < z_len:
                tok_input_ids.append(intermediate_tokens[i])
        tok_targets = list(tok_labels)

        return {
            'z_ids': torch.tensor(intermediate_tokens, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long),
            'del_labels': torch.tensor(del_labels, dtype=torch.long),
            'ins_labels': torch.tensor(ins_labels, dtype=torch.long),
            'tok_labels': torch.tensor(tok_labels, dtype=torch.long) if tok_labels else torch.zeros(0, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.float),
            'z_length': z_len,
            'target_length': len(target_tokens),
            'num_tok_labels': len(tok_labels),
            'tok_input_ids': torch.tensor(tok_input_ids, dtype=torch.long),
            'tok_positions': torch.tensor(tok_positions, dtype=torch.long) if tok_positions else torch.zeros(0, dtype=torch.long),
            'tok_targets': torch.tensor(tok_targets, dtype=torch.long) if tok_targets else torch.zeros(0, dtype=torch.long),
            'tok_input_length': len(tok_input_ids),
            'num_tok_positions': len(tok_positions),
        }

    def _fallback_item(self):
        """Minimal valid sample for error cases."""
        PAD = self.tok['pad']
        BOS = self.tok['bos']
        EOS = self.tok['eos']
        tokens = [BOS, EOS]
        z_len = len(tokens)
        return {
            'z_ids': torch.tensor(tokens, dtype=torch.long),
            'target_ids': torch.tensor(tokens, dtype=torch.long),
            'del_labels': torch.full((z_len,), -100, dtype=torch.long),
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


class LevTEditingCollator:
    """Dynamic padding collator (same logic as LevTCollator, scheme-agnostic)."""

    def __init__(self, pad_token_id, max_length=2048):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        max_z_len = min(max(item['z_length'] for item in batch), self.max_length)
        max_tgt_len = min(max(item['target_length'] for item in batch), self.max_length)
        max_ins_len = max_z_len + 1
        max_tok_len = max(max(item['num_tok_labels'] for item in batch), 1)
        max_tok_input_len = min(max(item['tok_input_length'] for item in batch), self.max_length)
        max_tok_pos_len = max(max(item['num_tok_positions'] for item in batch), 1)

        PAD = self.pad_token_id
        out = {k: [] for k in [
            'z_ids', 'attention_mask', 'del_labels', 'ins_labels',
            'tok_labels', 'context_mask', 'target_ids',
            'tok_input_ids', 'tok_attention_mask', 'tok_positions', 'tok_targets',
        ]}

        for item in batch:
            z_len = min(item['z_length'], max_z_len)
            tgt_len = min(item['target_length'], max_tgt_len)

            def _pad(tensor, max_l, pad_val):
                t = tensor[:max_l]
                if len(t) < max_l:
                    t = torch.cat([t, torch.full((max_l - len(t),), pad_val, dtype=t.dtype)])
                return t

            out['z_ids'].append(_pad(item['z_ids'], max_z_len, PAD))
            out['attention_mask'].append(
                torch.cat([torch.ones(z_len, dtype=torch.long),
                           torch.zeros(max_z_len - z_len, dtype=torch.long)])
                if z_len < max_z_len else torch.ones(z_len, dtype=torch.long)
            )
            out['del_labels'].append(_pad(item['del_labels'], max_z_len, -100))

            il = item['ins_labels'][:min(il_len := item['ins_labels'].size(0), max_ins_len)]
            if len(il) < max_ins_len:
                il = torch.cat([il, torch.zeros(max_ins_len - len(il), dtype=torch.long)])
            out['ins_labels'].append(il[:max_ins_len])

            out['tok_labels'].append(_pad(item['tok_labels'], max_tok_len, PAD))
            out['context_mask'].append(_pad(item['context_mask'], max_z_len, 0.0))
            out['target_ids'].append(_pad(item['target_ids'], max_tgt_len, PAD))

            tok_in_len = min(item['tok_input_length'], max_tok_input_len)
            out['tok_input_ids'].append(_pad(item['tok_input_ids'], max_tok_input_len, PAD))
            out['tok_attention_mask'].append(
                torch.cat([torch.ones(tok_in_len, dtype=torch.long),
                           torch.zeros(max_tok_input_len - tok_in_len, dtype=torch.long)])
                if tok_in_len < max_tok_input_len else torch.ones(tok_in_len, dtype=torch.long)
            )

            tp = item['tok_positions']
            tp_len = min(tp.size(0), max_tok_pos_len)
            if tp_len > 0:
                tp = tp[:tp_len]
                tp = tp[tp < max_tok_input_len]
                tp_len = tp.size(0)
            if tp_len < max_tok_pos_len:
                tp = torch.cat([tp, torch.full((max_tok_pos_len - tp_len,), -1, dtype=torch.long)])
            out['tok_positions'].append(tp[:max_tok_pos_len])

            out['tok_targets'].append(_pad(item['tok_targets'], max_tok_pos_len, PAD))

        return {k: torch.stack(v) for k, v in out.items()}


class BucketBatchSampler(Sampler):
    """Length-aware batch sampler (copied from dataset.py for independence)."""

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
    """Split npz files into train/val/test sets (same split as original)."""
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


def load_lengths_cache(data_dir, file_list, scheme='A'):
    """Try to load cached lengths or return defaults."""
    suffix = {'A': 'no_pair', 'B': 'no_pair_related', 'C': 'with_pair', 'D': 'absolute_bundled'}
    cache_path = os.path.join(data_dir, f'.lengths_cache_{suffix.get(scheme, "with_pair")}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_lengths = pickle.load(f)
        return [all_lengths.get(fname, 1500) for fname in file_list]
    return [1500] * len(file_list)
