"""
Music BERT MLM dataset (with_pair encoding scheme).

Reuses the with_pair tokenization logic and applies MLM masking on bundled tokens.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pickle
from typing import List, Dict
from my_tokenizer import PianoRollTokenizer
from config import MusicTokenConfig, BertPretrainConfig


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


class MLMDataset(Dataset):
    """MLM pretraining dataset (with_pair).

    Loads a piano roll from an npz file, encodes it into a with_pair
    bundled token sequence, then applies MLM masking on bundled tokens
    (0-7127).
    """

    def __init__(
        self,
        data_dir: str,
        token_config: MusicTokenConfig,
        bert_config: BertPretrainConfig,
        mode: str = 'train',
        cache_lengths: bool = True,
    ):
        self.data_dir = data_dir
        self.tc = token_config
        self.bc = bert_config
        self.mode = mode
        self.max_seq_len = bert_config.max_seq_len

        # Create the tokenizer
        self.tokenizer = PianoRollTokenizer(
            patch_h=token_config.patch_h,
            patch_w=token_config.patch_w,
            pattern_num=token_config.pattern_num,
            beats_length=token_config.beats_length,
        )

        # Load the file list
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        print(f"Found {len(self.data_files)} npz files")

        # Load the length cache
        cache_file = os.path.join(data_dir, '.lengths_cache_with_pair.pkl')
        self.file_lengths = None
        self.sorted_indices = None

        if cache_lengths and os.path.exists(cache_file):
            print("Loading length cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.data_files = cache_data['data_files']
            self.file_lengths = cache_data['lengths']
            self.sorted_indices = cache_data['sorted_indices']
            print(f"Cache loaded: {len(self.data_files)} files")
        elif cache_lengths:
            print(f"Warning: length cache not found ({cache_file}); length sorting disabled")

        # Split into train/test
        self._split_data()

        # Set of special tokens
        self._special_ids = token_config.special_token_ids

    def _split_data(self):
        """Split into train/test at the song level."""
        total = len(self.data_files)
        rng = np.random.RandomState(self.bc.random_seed)
        indices = np.arange(total)
        rng.shuffle(indices)

        test_size = int(total * self.bc.test_split_ratio)
        train_size = total - test_size

        if self.mode == 'train':
            selected = indices[:train_size]
            print(f"Training set: {len(selected)} files")
        elif self.mode == 'test':
            selected = indices[train_size:]
            print(f"Test set: {len(selected)} files")
        else:
            selected = indices
            print(f"All data: {len(selected)} files")

        self.data_files = [self.data_files[i] for i in selected]
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in selected]
            self.sorted_indices = sorted(
                range(len(self.file_lengths)),
                key=lambda i: self.file_lengths[i]
            )

    def __len__(self):
        return len(self.data_files)

    def _tokenize_npz(self, idx):
        """Encode an npz file into a with_pair bundled token sequence."""
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        file_name = self.data_files[idx]
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()

        time_sig_idx = metadata['time_signature_idx']
        if time_sig_idx == 9:
            time_sig_idx = 4

        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        is_continuation = metadata.get('is_continuation', False)

        # Decide whether to add BOS
        add_bos = True
        if '_' in file_name:
            suffix = file_name.split('_')[-1].replace('.npz', '')
            if suffix.isdigit() and suffix != '1':
                add_bos = False

        # Random pitch shift
        pitch_shift = 0
        if np.random.random() < 0.7:
            pitch_shift = np.random.randint(-5, 6)

        # Process each bar
        all_tokens = []

        # BOS + TIME_SIG + BPM
        if add_bos:
            all_tokens.append(self.tc.bos_token_id)
        all_tokens.append(self.tc.time_sig_offset_id + time_sig_idx)
        all_tokens.append(self.tc.bpm_offset_id + encode_bpm(bpm_value))

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
            beat_length = self.tc.patch_w
            num_beats = (t + beat_length - 1) // beat_length

            # BAR token
            all_tokens.append(self.tc.bar_token_id)

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

                # Track 0 (upper voice) - with_pair uses SPLIT_0
                p0 = beat_measure[:2]
                tok0 = self.tokenizer.image_to_patch_tokens(p0, strict_mode=True)
                comp0 = self.tokenizer.compress_tokens(
                    tok0,
                    split_marker_id=self.tc.split_0_id,
                    empty_marker_id=self.tc.empty_marker_id,
                )
                all_tokens.extend(comp0.tolist())

                # Track 1 (lower voice) - with_pair uses SPLIT_1
                p1 = beat_measure[2:]
                tok1 = self.tokenizer.image_to_patch_tokens(p1, strict_mode=True)
                comp1 = self.tokenizer.compress_tokens(
                    tok1,
                    split_marker_id=self.tc.split_1_id,
                    empty_marker_id=self.tc.empty_marker_id,
                )
                all_tokens.extend(comp1.tolist())

        # EOS
        if not is_continuation:
            all_tokens.append(self.tc.eos_token_id)

        return torch.tensor(all_tokens, dtype=torch.long)

    def _truncate(self, tokens):
        """Overlength truncation: take a random segment."""
        seq_len = len(tokens)
        if seq_len <= self.max_seq_len:
            return tokens

        prob = np.random.random()
        if prob < 0.33:
            return tokens[:self.max_seq_len]
        elif prob < 0.66:
            return tokens[-self.max_seq_len:]
        else:
            start = np.random.randint(0, seq_len - self.max_seq_len + 1)
            return tokens[start:start + self.max_seq_len]

    def _apply_mlm_mask(self, tokens):
        """Apply MLM masking on bundled tokens.

        Only bundled tokens (0-7127) are masked; control tokens (7128+) are
        not.

        Returns:
            input_ids: the masked sequence.
            labels: original token at masked positions, -100 elsewhere.
        """
        input_ids = tokens.clone()
        labels = torch.full_like(tokens, -100)

        # Determine which positions are bundled tokens (0-7127)
        is_note = (tokens >= self.tc.note_token_min) & (tokens <= self.tc.note_token_max)

        # Randomly select 15% of the bundled tokens
        note_indices = torch.where(is_note)[0]
        if len(note_indices) == 0:
            return input_ids, labels

        num_to_mask = max(1, int(len(note_indices) * self.bc.mask_prob))
        perm = torch.randperm(len(note_indices))[:num_to_mask]
        mask_indices = note_indices[perm]

        # Record labels
        labels[mask_indices] = tokens[mask_indices]

        # 80% -> [MASK]
        rand = torch.rand(num_to_mask)
        mask_replace = mask_indices[rand < self.bc.mask_replace_prob]
        input_ids[mask_replace] = self.tc.mask_token_id

        # 10% -> random bundled token (0-7127)
        mask_random = mask_indices[
            (rand >= self.bc.mask_replace_prob) &
            (rand < self.bc.mask_replace_prob + self.bc.mask_random_prob)
        ]
        random_tokens = torch.randint(
            self.tc.note_token_min, self.tc.note_token_max + 1,
            (len(mask_random),), dtype=torch.long
        )
        input_ids[mask_random] = random_tokens

        # Remaining 10% kept unchanged (labels already set, input_ids untouched)

        return input_ids, labels

    def __getitem__(self, idx):
        tokens = self._tokenize_npz(idx)
        tokens = self._truncate(tokens)
        input_ids, labels = self._apply_mlm_mask(tokens)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class BucketBatchSampler(Sampler):
    """Length-aware batch sampler."""

    def __init__(self, dataset, batch_size=64, bucket_size=200, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        if dataset.sorted_indices is None:
            self.buckets = [list(range(len(dataset)))]
        else:
            self.buckets = []
            for i in range(0, len(dataset.sorted_indices), bucket_size):
                self.buckets.append(dataset.sorted_indices[i:i + bucket_size])

        print(f"Created {len(self.buckets)} length buckets")

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
        return sum(len(b) for b in self.buckets) // self.batch_size


class MLMCollator:
    """MLM data collator: dynamic padding."""

    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(f['input_ids']) for f in features),
            self.max_length
        )

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            ids = f['input_ids']
            lab = f['labels']
            seq_len = len(ids)

            if seq_len < max_len:
                pad_len = max_len - seq_len
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
                attn = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            elif seq_len > max_len:
                ids = ids[:max_len]
                lab = lab[:max_len]
                attn = torch.ones(max_len, dtype=torch.long)
            else:
                attn = torch.ones(seq_len, dtype=torch.long)

            batch_input_ids.append(ids)
            batch_labels.append(lab)
            batch_attention_mask.append(attn)

        return {
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_attention_mask),
        }
