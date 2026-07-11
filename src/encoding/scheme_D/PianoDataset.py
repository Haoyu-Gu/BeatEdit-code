# dataset.py - full code

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
            return 0  # slow
        elif bpm <= 200:
            return 1  # medium
        else:
            return 2  # fast



class PianoDataset(Dataset):
    """Length-aware dataset."""

    def __init__(self, data_dir, config, cache_lengths=True, mode='train',
                 test_split_ratio=0.05, random_seed=42):
        """
        Args:
            data_dir: data directory.
            config: model configuration.
            cache_lengths: whether to use the length cache.
            mode: 'train' or 'test', selecting the training or test split.
            test_split_ratio: test-set fraction (between 0 and 1).
            random_seed: random seed for a reproducible split.
        """
        self.root_dir = data_dir

        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.max_seq_len = config.train_cutoff_len

        self.pad_token = config.pad_token_id
        self.bos_token = config.bos_token_id
        self.eos_token = config.eos_token_id
        self.bar_token = config.bar_token_id

        self.split_0_id = config.split_0_id
        self.split_1_id = config.split_1_id
        self.empty_marker_id = config.empty_marker_id

        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id

        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed

        self.config = config

        # Create the tokenizer instance
        self.tokenizer = PianoRollTokenizer(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            pattern_num=config.pattern_num,
            beats_length=config.beats_length,
        )

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        print(f"Found {len(self.data_files)} valid npz files")

        # Precompute length information
        cache_file = os.path.join(data_dir, '.lengths_cache.pkl')

        if cache_lengths and os.path.exists(cache_file):
            print("Loading length cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify that the patch parameters match
            if (cache_data['patch_h'] != self.patch_h or
                cache_data['patch_w'] != self.patch_w):
                raise ValueError(
                    f"Cached patch parameters ({cache_data['patch_h']}x{cache_data['patch_w']}) "
                    f"do not match the config ({self.patch_h}x{self.patch_w}); please re-run precompute_lengths.py"
                )

            self.data_files = cache_data['data_files']
            self.file_lengths = cache_data['lengths']
            self.sorted_indices = cache_data['sorted_indices']

            print(f"Loaded length information for {len(self.data_files)} files")

        elif cache_lengths:
            raise FileNotFoundError(
                f"Length cache not found: {cache_file}\n"
                f"Please run first: python precompute_lengths.py"
            )
        else:
            # No cache, traditional path
            self.file_lengths = None
            self.sorted_indices = None
            print(f"Found {len(self.data_files)} files (length cache not used)")

        # Split into train and test
        self._split_train_test()

    def _split_train_test(self):
        """Split into train and test according to the mode argument."""
        total_files = len(self.data_files)

        # Set the random seed for reproducibility
        rng = np.random.RandomState(self.random_seed)

        # Create an index array and shuffle it
        indices = np.arange(total_files)
        rng.shuffle(indices)  # only affects the shuffle here


        # Compute the test-set size
        test_size = int(total_files * self.test_split_ratio)
        train_size = total_files - test_size

        if self.config.min_length > 0:
            # Filter samples by length
            filtered_indices = []
            for i in indices:
                length = self.file_lengths[i]
                if length > self.config.min_length:
                    filtered_indices.append(i)
            indices = np.array(filtered_indices)
            total_files = len(indices)
            test_size = int(total_files * self.test_split_ratio)
            train_size = total_files - test_size
            print(f"{total_files} files remaining after filtering")

        if self.mode == 'train':
            # Use the first train_size samples as the training set
            selected_indices = indices[:train_size]
            print(f"Using training set: {len(selected_indices)} files ({train_size}/{total_files})")
        elif self.mode == 'test':
            # Use the last test_size samples as the test set
            selected_indices = indices[train_size:]
            print(f"Using test set: {len(selected_indices)} files ({test_size}/{total_files})")
        elif self.mode == 'all':
            # Use all data
            selected_indices = indices
            print(f"Using all data: {len(selected_indices)} files ({total_files}/{total_files})")
        elif self.mode == 'toy':
            # Use a small subset
            selected_indices = indices[:4000]
            print(f"Using validation data: {len(selected_indices)} files ({4000}/{total_files})")
        else:
            raise ValueError(f"mode must be 'train', 'test', or 'all'; got: {self.mode}")

        # Update data_files and the related indices
        self.data_files = [self.data_files[i] for i in selected_indices]

        # If the length cache was used, update the related info too
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in selected_indices]

            # Rebuild sorted_indices (ordering within the new subset)
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
        """Process the upper and lower parts interleaved at the beat level.

        Args:
            measure: (4, 88, t) - first 2 channels are part0 (upper voice),
                last 2 channels are part1 (lower voice).

        Returns:
            part0_beat_tokens: beat-level token list for part0.
            part1_beat_tokens: beat-level token list for part1.
        """
        part0_beat_tokens = []
        part1_beat_tokens = []

        t = measure.shape[2]
        beat_length = self.patch_w

        # Compute the actual number of beats
        num_beats = (t + beat_length - 1) // beat_length  # round up

        for beat_idx in range(num_beats):
            # Compute the time range of the current beat
            start_t = beat_idx * beat_length
            end_t = min(start_t + beat_length, t)

            # Pad if the last beat is not long enough
            beat_measure = measure[:, :, start_t:end_t]
            current_length = end_t - start_t

            if current_length < beat_length:
                # Pad to a full beat length
                pad_width = ((0, 0), (0, 0), (0, beat_length - current_length))
                beat_measure = np.pad(beat_measure, pad_width, mode='constant', constant_values=0)

            # === Process part0 (upper voice) ===
            part0_beat = beat_measure[:2]  # (2, 88, beat_length)
            tokens_0 = self.tokenizer.image_to_patch_tokens(part0_beat, strict_mode=True)
            compressed_tokens_0 = self.tokenizer.compress_tokens(
                tokens_0, split_marker_id=self.split_0_id, empty_marker_id=self.empty_marker_id)
            part0_beat_tokens.append(torch.tensor(compressed_tokens_0, dtype=torch.long))

            # === Process part1 (lower voice) ===
            part1_beat = beat_measure[2:]  # (2, 88, beat_length)
            tokens_1 = self.tokenizer.image_to_patch_tokens(part1_beat, strict_mode=True)
            compressed_tokens_1 = self.tokenizer.compress_tokens(
                tokens_1, split_marker_id=self.split_1_id, empty_marker_id=self.empty_marker_id)
            part1_beat_tokens.append(torch.tensor(compressed_tokens_1, dtype=torch.long))

        return part0_beat_tokens, part1_beat_tokens


    def _interleave_tokens(self, part0_beats, part1_beats):
        """Interleave and concatenate the tokens of part0 and part1.

        Args:
            part0_beats: list of part0 beat tokens [tensor0, tensor0, ...].
            part1_beats: list of part1 beat tokens.

        Returns:
            The interleaved, concatenated tensor.
        """
        tokens = []
        for p0, p1 in zip(part0_beats, part1_beats):
            tokens.append(p0)
            tokens.append(p1)
        return torch.cat(tokens, dim=0)


    def __getitem__(self, idx):
        # Load the data
        file_path = os.path.join(self.root_dir, self.data_files[idx])
        file_name = self.data_files[idx]
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()

        # Extract metadata
        time_sig_idx = metadata['time_signature_idx']
        if time_sig_idx == 9:
            time_sig_idx = 4  # handle a special time signature

        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        is_continuation = metadata.get('is_continuation', False)

        # Decide whether to add BOS (based on the file name)
        add_bos = True
        if '_' in file_name:
            suffix = file_name.split('_')[-1].replace('.npz', '')
            if suffix.isdigit() and suffix != '1':
                add_bos = False

        # Random pitch shift (data augmentation)
        pitch_shift = 0
        if np.random.random() < 0.7:
            pitch_shift = np.random.randint(-5, 6)

        # Process all measures and collect tokens
        measure_tokens = []
        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']
            measure = measure[:, ::-1, :].copy()
            # Apply the pitch shift
            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            # Process beats and interleave
            part0_beats, part1_beats = self._process_measure_with_beat_interleaving(measure)
            beat_tokens = self._interleave_tokens(part0_beats, part1_beats)

            # Add the bar token and the beat tokens
            measure_tokens.append(torch.tensor([self.bar_token], dtype=torch.long))
            measure_tokens.append(beat_tokens)

        # Build the full sequence
        tokens = []
        labels = []

        # Add BOS
        if add_bos:
            tokens.append(torch.tensor([self.bos_token], dtype=torch.long))
            labels.append(torch.tensor([-100], dtype=torch.long))

        # Add time signature and BPM
        time_sig_token = time_sig_idx + self.time_sig_offset_id
        bpm_token = encode_bpm(bpm_value) + self.bpm_offset_id
        tokens.append(torch.tensor([time_sig_token, bpm_token], dtype=torch.long))
        labels.append(torch.tensor([-100, -100], dtype=torch.long))

        # Add the musical content
        content = torch.cat(measure_tokens)
        tokens.append(content)
        labels.append(content)

        # Add EOS
        if not is_continuation:
            tokens.append(torch.tensor([self.eos_token], dtype=torch.long))
            labels.append(torch.tensor([self.eos_token], dtype=torch.long))

        # Concatenate all tokens
        input_ids = torch.cat(tokens)
        labels = torch.cat(labels)

        # Overlength truncation: take a random segment; the first 8% is not counted in the loss
        seq_len = len(input_ids)

        if seq_len > self.max_seq_len :
            prob = np.random.random()
            if  prob < 0.25:
                # Truncate from the start
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]
            elif prob < 0.5:
                # Truncate from the end
                input_ids = input_ids[-self.max_seq_len:]
                labels = labels[-self.max_seq_len:]

                # The first 8% is not counted in the loss
                ignore_len = int(self.max_seq_len * 0.08)
                labels[:ignore_len] = -100

            else:
                start_idx = np.random.randint(0, seq_len - self.max_seq_len + 1)
                input_ids = input_ids[start_idx:start_idx + self.max_seq_len]
                labels = labels[start_idx:start_idx + self.max_seq_len]

                # The first 8% is not counted in the loss
                ignore_len = int(self.max_seq_len * 0.08)
                labels[:ignore_len] = -100



        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class BucketBatchSampler(Sampler):
    """Length-aware batch sampler."""

    def __init__(self, dataset, batch_size=16, bucket_size=100, shuffle=True):
        """
        Args:
            dataset: PianoDataset instance.
            batch_size: actual training batch size.
            bucket_size: size of each length bucket.
            shuffle: whether to randomize.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        if dataset.sorted_indices is None:
            raise ValueError("Dataset needs cache_lengths=True")

        self._create_buckets()

    def _create_buckets(self):
        """Group samples of similar length into buckets."""
        self.buckets = []
        sorted_indices = self.dataset.sorted_indices

        # Split the sorted indices into buckets
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size]
            self.buckets.append(bucket)

        print(f"Created {len(self.buckets)} length buckets")

    def __iter__(self):
        """Yield batch indices."""
        # Randomly shuffle the order of the buckets
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))

        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()

            # Shuffle within the bucket
            if self.shuffle:
                np.random.shuffle(bucket)

            # Produce batches from the bucket
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:  # make sure the batch is not empty
                    yield batch

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets) // self.batch_size


@dataclass
class DataCollatorForVariableLengthLM:
    """Data collator supporting dynamic padding."""

    def __init__(self, config):
        self.pad_token_id = config.pad_token_id
        self.max_length = config.train_cutoff_len

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get the maximum length in the batch
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