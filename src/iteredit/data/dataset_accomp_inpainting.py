"""
Dataset for Levenshtein Transformer Accompaniment-Only Inpainting.

Unlike vanilla inpainting (masks entire beats), this dataset:
- Keeps melody tokens intact in the masked region
- Only removes accompaniment tokens
- Trains the model to generate accomp conditioned on melody context

This matches the evaluation T2 inpainting test format.

Flow per sample:
1. Tokenize NPZ → target_tokens (full sequence)
2. Parse → separate melody/accomp beats
3. Select contiguous beat range → remove accomp tokens (create source)
4. Corrupt target accomp tokens → create intermediate state
5. Compute per-beat Levenshtein labels (intermediate → target)
6. Return tensors for LevT three-head training
"""

import os
import sys
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset

LEVT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIOUS_DIR = os.path.dirname(LEVT_DIR)
UNIFIED_DIR = os.path.join(PREVIOUS_DIR, 'evaluation')
sys.path.insert(0, LEVT_DIR)
sys.path.insert(0, UNIFIED_DIR)

from scheme_utils import SchemeLoader
from data.levenshtein_utils import compute_edit_labels
from data.dataset_editing import (
    SCHEME_TOKENS, DATA_DIR, LevTEditingCollator, BucketBatchSampler,
    get_file_lists, load_lengths_cache,
)


class LevTAccompInpaintingDataset(Dataset):
    """
    Dataset for accomp-only inpainting training.

    Selects a contiguous range of accompaniment beats, removes their tokens
    (keeping melody), then corrupts the target accomp as intermediate state
    for Levenshtein label computation.
    """

    def __init__(
        self,
        file_list,
        scheme='A',
        data_dir=DATA_DIR,
        max_len=2048,
        mask_ratio_min=0.15,
        mask_ratio_max=0.50,
        corruption_delete_prob=0.3,
        corruption_replace_prob=0.2,
        pitch_shift_augment=False,
        max_insert=20,
    ):
        self.file_list = file_list
        self.scheme = scheme
        self.data_dir = data_dir
        self.max_len = max_len
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.corruption_delete_prob = corruption_delete_prob
        self.corruption_replace_prob = corruption_replace_prob
        self.pitch_shift_augment = pitch_shift_augment
        self.max_insert = max_insert

        self.tok = SCHEME_TOKENS[scheme]

        self._loader = SchemeLoader(scheme)
        felix = self._loader.felix
        self._felix_ds = felix._FELIXBaseDataset(
            file_list=file_list,
            data_dir=data_dir,
            max_len=4096,
            pitch_shift_augment=pitch_shift_augment,
        )
        self._parse_fn = felix._parse_sequence
        self._separate_fn = felix._separate_tracks
        self._rebuild_fn = felix._rebuild_interleaved

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
        """Corrupt beat tokens to simulate intermediate refinement state."""
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

    def _select_mask_range(self, n_beats):
        """Select a contiguous range of beats to mask."""
        ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        mask_len = max(1, int(n_beats * ratio))
        mask_len = min(mask_len, n_beats)
        start = random.randint(0, n_beats - mask_len)
        return start, start + mask_len

    def _remove_accomp_tokens(self, mel_beats, acc_beats, parsed, mask_start, mask_end):
        """Remove accomp tokens from masked beats, keep melody.

        Creates source by setting masked accomp beats to empty tokens.
        """
        source_acc = copy.deepcopy(acc_beats)
        for i in range(mask_start, min(mask_end, len(source_acc))):
            source_acc[i] = copy.deepcopy(acc_beats[i])
            source_acc[i]['tokens'] = []  # remove all accomp tokens
        return self._rebuild_fn(mel_beats, source_acc, parsed)

    def _get_item_impl(self, idx):
        PAD = self.tok['pad']
        PLH = self.tok['plh']

        # 1. Tokenize
        target_tokens = self._felix_ds._tokenize_npz(idx)
        target_tokens = self._truncate(target_tokens)

        # 2. Parse and separate
        parsed = self._parse_fn(target_tokens)
        mel_beats, acc_beats = self._separate_fn(parsed)

        if len(acc_beats) < 4:
            return self._fallback_item()

        # 3. Select contiguous mask range
        mask_start, mask_end = self._select_mask_range(len(acc_beats))
        changed_mask = [False] * len(acc_beats)
        for i in range(mask_start, mask_end):
            if i < len(acc_beats) and len(acc_beats[i]['tokens']) > 0:
                changed_mask[i] = True

        if not any(changed_mask):
            return self._fallback_item()

        # 4. Create intermediate: target with masked accomp beats corrupted
        # (NOT removed — corrupted, so the model learns to refine)
        inter_acc = copy.deepcopy(acc_beats)
        for i in range(len(changed_mask)):
            if changed_mask[i]:
                inter_acc[i]['tokens'] = self._corrupt_tokens(acc_beats[i]['tokens'])

        intermediate_tokens = self._rebuild_fn(mel_beats, inter_acc, parsed)
        if len(intermediate_tokens) > self.max_len:
            intermediate_tokens = intermediate_tokens[:self.max_len]

        # 5. Re-parse intermediate and target
        inter_parsed = self._parse_fn(intermediate_tokens)
        _, inter_acc_beats = self._separate_fn(inter_parsed)

        tgt_parsed = self._parse_fn(target_tokens)
        _, tgt_acc_beats = self._separate_fn(tgt_parsed)

        # 6. Compute per-beat Levenshtein labels
        z_len = len(intermediate_tokens)
        del_labels = [-100] * z_len
        ins_labels = [0] * (z_len + 1)
        tok_labels = []
        context_mask = [0.0] * z_len

        n_beats = min(len(changed_mask), len(inter_acc_beats), len(tgt_acc_beats))

        for i in range(n_beats):
            if not changed_mask[i]:
                continue

            inter_beat = inter_acc_beats[i]
            tgt_beat = tgt_acc_beats[i]

            z_beat = inter_beat['tokens']
            y_beat = tgt_beat['tokens']

            if inter_beat['start_idx'] >= z_len:
                continue

            beat_del, beat_ins, beat_tok = compute_edit_labels(z_beat, y_beat)

            note_start = inter_beat['start_idx'] + 1

            for j, d in enumerate(beat_del):
                pos = note_start + j
                if pos < z_len:
                    del_labels[pos] = d
                    context_mask[pos] = 1.0

            for j, n in enumerate(beat_ins):
                pos = note_start + j
                if pos <= z_len:
                    ins_labels[pos] = n

            tok_labels.extend(beat_tok)

        # 7. Clamp ins_labels
        ins_labels = [min(x, self.max_insert) for x in ins_labels]

        # 8. Build tok_input_ids
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
