"""
Piano Roll Tokenizer - piano-roll encoder/decoder.

Centralizes the encode/decode logic between a piano roll and tokens, including:
- Patch-based tokenization (ternary encoding)
- Relative-position bundled compression encoding
- Dual-channel piano-roll reconstruction

Author: Optimized from original implementation
"""

import numpy as np
import torch
from typing import Optional, Union


class PianoRollTokenizer:
    """Token encoder/decoder for a piano roll.

    Responsibilities:
    1. Convert a dual-channel piano roll (sustain + onset) into patch tokens
       (ternary convention matches the paper: 0 = silent, 1 = onset,
       2 = sustain continuation).
    2. Compress the token sequence using relative-position bundled encoding.
    3. Support a full encode-decode round-trip.

    Args:
        patch_h: patch height (default 2, i.e. 2 pitches).
        patch_w: patch width (default 4, i.e. 4 time steps).
        pattern_num: multiplier used when bundling relative position and
            pattern value into a single token (default 81).
        beats_length: number of pitches per beat / image height, i.e. the
            88 piano keys (default 88).

    Example:
        >>> tokenizer = PianoRollTokenizer(patch_h=2, patch_w=4)
        >>> # encode
        >>> image = np.random.randint(0, 2, (2, 88, 16))
        >>> compressed = tokenizer.encode(image)
        >>> # decode
        >>> reconstructed = tokenizer.decode(compressed, num_beats=4)
    """

    def __init__(
        self,
        patch_h: int = 1,
        patch_w: int = 4,
        pattern_num: int = 81,
        beats_length: int = 88,
        use_velocities: bool = False,
        velocity_offset: Optional[int] = None,
        num_velocities: Optional[int] = None,
    ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.pattern_num = pattern_num
        self.beats_length = beats_length
        self.use_velocities = use_velocities

        # Velocity-related parameters
        if use_velocities:
            if velocity_offset is None:
                raise ValueError("velocity_offset must be provided when use_velocities=True")
            if num_velocities is None:
                raise ValueError("num_velocities must be provided when use_velocities=True")
            self.velocity_offset = velocity_offset
            self.num_velocities = num_velocities
        else:
            self.velocity_offset = velocity_offset
            self.num_velocities = num_velocities

        # Precompute patch-related parameters
        self.patch_size = patch_h * patch_w
        self.powers_3 = 3 ** np.arange(self.patch_size - 1, -1, -1)

        # Special-token replacement rules (used in strict mode)
        self.special_token_ids = [
            26, 24, 34, 62, 47, 19, 29, 38, 74, 60,
            56, 7, 21, 65, 23, 22, 20, 25, 61, 11,
            6, 18, 2, 33, 8
        ]
        self.replacement_ids = [0, 53, 5, 80, 45]

    def quantize_velocity(self, velocity: Union[np.ndarray, int, float]) -> Union[np.ndarray, int]:
        """Quantize a velocity from 1-127 into the range 1-num_velocities.

        Args:
            velocity: original velocity value (1-127), scalar or array.

        Returns:
            Quantized velocity (1-num_velocities); 0 stays 0 (no note).
        """
        if self.num_velocities is None:
            raise ValueError("num_velocities not set")

        if isinstance(velocity, np.ndarray):
            result = np.zeros_like(velocity, dtype=np.int64)
            non_zero_mask = velocity > 0
            # Map 1-127 to 1-num_velocities
            # Linear quantization: q = ceil((v / 127) * num_velocities)
            result[non_zero_mask] = np.ceil(
                velocity[non_zero_mask] / 127.0 * self.num_velocities
            ).astype(np.int64)
            # Clamp to the range 1-num_velocities
            result[non_zero_mask] = np.clip(result[non_zero_mask], 1, self.num_velocities)
            return result
        else:
            if velocity == 0:
                return 0
            q = int(np.ceil(velocity / 127.0 * self.num_velocities))
            return max(1, min(self.num_velocities, q))

    def dequantize_velocity(self, quantized: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """Restore a quantized velocity (1-num_velocities) back to 1-127.

        Args:
            quantized: quantized velocity (1-num_velocities).

        Returns:
            Restored velocity (1-127); 0 stays 0 (no note).
        """
        if self.num_velocities is None:
            raise ValueError("num_velocities not set")

        if isinstance(quantized, np.ndarray):
            result = np.zeros_like(quantized, dtype=np.int64)
            non_zero_mask = quantized > 0
            # Dequantization: v = round((q / num_velocities) * 127)
            result[non_zero_mask] = np.round(
                quantized[non_zero_mask] / self.num_velocities * 127.0
            ).astype(np.int64)
            # Clamp to the range 1-127
            result[non_zero_mask] = np.clip(result[non_zero_mask], 1, 127)
            return result
        else:
            if quantized == 0:
                return 0
            v = int(np.round(quantized / self.num_velocities * 127.0))
            return max(1, min(127, v))

    def encode(
        self,
        image: Union[np.ndarray, torch.Tensor],
        split_marker_id: int,
        empty_marker_id: int,
        use_strict_mode: bool = True
    ) -> np.ndarray:
        """Full encoding pipeline: piano roll -> compressed tokens (bundled encoding).

        Args:
            image: dual-channel piano roll of shape (2, 88, t);
                ch0 = sustain, ch1 = onset.
            split_marker_id: voice split-marker ID.
            empty_marker_id: empty-segment marker ID.
            use_strict_mode: whether to use strict mode (replace special tokens).

        Returns:
            compressed_sequence: 1-D compressed token sequence.
        """
        tokens = self.image_to_patch_tokens(image, strict_mode=use_strict_mode)
        compressed = self.compress_tokens(tokens, split_marker_id=split_marker_id, empty_marker_id=empty_marker_id)
        return compressed

    def decode(
        self,
        compressed_sequence: Union[np.ndarray, list],
        split_marker_id: int,
        empty_marker_id: int
    ) -> np.ndarray:
        """Full decoding pipeline: compressed tokens -> piano roll (bundled encoding).

        Args:
            compressed_sequence: compressed token sequence.
            split_marker_id: voice split-marker ID.
            empty_marker_id: empty-segment marker ID.

        Returns:
            image: dual-channel piano roll of shape (2, 88, t).
        """
        tokens = self.decompress_tokens(compressed_sequence, split_marker_id=split_marker_id, empty_marker_id=empty_marker_id)
        image = self.patch_tokens_to_image(tokens)
        return image

    # ==================== Encoding methods ====================

    def image_to_patch_tokens(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True
    ) -> np.ndarray:
        """Convert a dual-channel piano roll into patch tokens (ternary pattern).

        Args:
            image: dual-channel piano roll of shape (2, 88, t);
                ch0 = sustain, ch1 = onset.
            strict_mode: whether to replace special tokens.

        Returns:
            tokens: token matrix of shape (num_time_patches, num_pitch_patches).

        Encoding convention:
            0 = silent (sustain=0, onset=0)
            1 = onset (sustain=1, onset=1)
            2 = sustain continuation (sustain=1, onset=0)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Ensure the input has two channels
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain_channel = image[0].copy()  # shape: (88, t)
        onset_channel = image[1].copy()    # shape: (88, t)

        # An onset can only occur where sustain is 1
        onset_channel[sustain_channel == 0] = 0

        beats_length, img_w = sustain_channel.shape

        # Pad the width so it is divisible by patch_w
        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain_channel = np.pad(
                sustain_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            onset_channel = np.pad(
                onset_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            img_w = sustain_channel.shape[1]

        num_patch_rows = beats_length // self.patch_h
        num_patch_cols = img_w // self.patch_w

        # Reshape into patches
        sustain_patches = self._reshape_to_patches(sustain_channel, num_patch_rows, num_patch_cols)
        onset_patches = self._reshape_to_patches(onset_channel, num_patch_rows, num_patch_cols)

        # Combine into the ternary encoding
        combined_patches = 2 * sustain_patches.astype(np.int64) - onset_patches.astype(np.int64)

        # Compute token values in base 3
        tokens = np.dot(combined_patches, self.powers_3)

        # Handle special tokens (strict mode)
        if strict_mode:
            tokens = self._replace_special_tokens(tokens)

        return tokens

    def _reshape_to_patches(
        self,
        channel: np.ndarray,
        num_patch_rows: int,
        num_patch_cols: int
    ) -> np.ndarray:
        """Reshape a channel into patches."""
        patches = channel.reshape(num_patch_rows, self.patch_h, num_patch_cols, self.patch_w)
        patches = patches.transpose(2, 0, 1, 3)  # (cols, rows, h, w)
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_size)
        return patches

    def _replace_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Randomly replace special tokens."""
        mask = np.isin(tokens, self.special_token_ids)
        if np.any(mask):
            num_replacements = np.sum(mask)
            random_replacements = np.random.choice(self.replacement_ids, size=num_replacements)
            tokens = tokens.copy()
            tokens[mask] = random_replacements
        return tokens

    def image_patch_tokens_velocity(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True
    ) -> np.ndarray:
        """Convert a dual-channel piano roll into velocity-aware patch tokens.

        Args:
            image: dual-channel piano roll of shape (2, 88, t);
                ch0 = sustain (0-127 velocity), ch1 = onset (0-127 velocity).
            strict_mode: whether to replace special tokens.

        Returns:
            tokens: dual-channel token matrix of shape
                (2, num_time_patches, num_pitch_patches);
                ch0 = ternary-encoded note pattern,
                ch1 = quantized velocity (0-num_velocities); 0 = no note,
                1-num_velocities = quantized velocity.

        Encoding convention:
            channel 0 (pattern):
                0 = silent (sustain=0, onset=0)
                1 = onset (sustain>0, onset>0)
                2 = sustain continuation (sustain>0, onset=0)
            channel 1 (velocity):
                0 = no note
                1-num_velocities = quantized mean velocity of nonzero values
                in the patch
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Ensure the input has two channels
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain_channel = image[0].copy()  # shape: (88, t), velocity values 0-127
        onset_channel = image[1].copy()    # shape: (88, t), velocity values 0-127

        # An onset can only occur where sustain is nonzero
        onset_channel[sustain_channel == 0] = 0

        # Build binary masks for pattern encoding
        sustain_binary = (sustain_channel > 0).astype(np.int64)
        onset_binary = (onset_channel > 0).astype(np.int64)

        beats_length, img_w = sustain_channel.shape

        # Pad the width so it is divisible by patch_w
        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain_binary = np.pad(
                sustain_binary,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            onset_binary = np.pad(
                onset_binary,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            sustain_channel = np.pad(
                sustain_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            img_w = sustain_binary.shape[1]

        num_patch_rows = beats_length // self.patch_h
        num_patch_cols = img_w // self.patch_w

        # Reshape into patches (for pattern encoding)
        sustain_patches = self._reshape_to_patches(sustain_binary, num_patch_rows, num_patch_cols)
        onset_patches = self._reshape_to_patches(onset_binary, num_patch_rows, num_patch_cols)

        # Combine into the ternary encoding
        combined_patches = 2 * sustain_patches.astype(np.int64) - onset_patches.astype(np.int64)

        # Compute pattern token values in base 3
        pattern_tokens = np.dot(combined_patches, self.powers_3)

        # Handle special tokens (strict mode)
        if strict_mode:
            pattern_tokens = self._replace_special_tokens(pattern_tokens)

        # Compute the velocity channel
        # Reshape sustain_channel into patches to compute the mean velocity
        sustain_velocity_patches = self._reshape_to_patches(sustain_channel, num_patch_rows, num_patch_cols)

        # Compute the mean velocity of nonzero values within each patch
        velocity_tokens = np.zeros((num_patch_cols, num_patch_rows), dtype=np.int64)

        for col in range(num_patch_cols):
            for row in range(num_patch_rows):
                patch_values = sustain_velocity_patches[col, row]  # shape: (patch_size,)
                non_zero_values = patch_values[patch_values > 0]
                if len(non_zero_values) > 0:
                    # Compute the mean and round it
                    avg_velocity = np.round(np.mean(non_zero_values)).astype(np.int64)
                    # Quantize the velocity into the range 1-num_velocities
                    velocity_tokens[col, row] = self.quantize_velocity(avg_velocity)
                else:
                    velocity_tokens[col, row] = 0

        # Stack into dual-channel output: (2, num_time_patches, num_pitch_patches)
        tokens = np.stack([pattern_tokens, velocity_tokens], axis=0)

        return tokens

    def compress_tokens(
        self,
        token_matrix: np.ndarray,
        split_marker_id: int,
        empty_marker_id: int
    ) -> np.ndarray:
        """Compress a token sequence using bundled encoding (relative position x pattern_num + patch value).

        Args:
            token_matrix: token matrix of shape (num_beats, beats_length).
            split_marker_id: voice split-marker ID (SPLIT_0 or SPLIT_1).
            empty_marker_id: empty-segment marker ID.

        Returns:
            compressed_sequence: compressed 1-D sequence.

        Encoding format:
            non-empty: [split_marker] [bundled_0] [bundled_1] ...
                       bundled = relative_position * pattern_num + token_value
            empty:     [empty_marker]
        """
        compressed_sequences = []

        for beat_tokens in token_matrix:
            # Find the positions of all nonzero tokens
            non_zero_indices = np.where(beat_tokens != 0)[0]

            if len(non_zero_indices) == 0:
                # Empty beat
                compressed = [empty_marker_id]
            else:
                # Non-empty beat - start with the split marker
                compressed = [split_marker_id]

                # Bundle each note using its relative position
                prev_pos = 0
                for idx in non_zero_indices:
                    relative_pos = idx - prev_pos
                    token_value = int(beat_tokens[idx])
                    bundled = int(relative_pos * self.pattern_num + token_value)
                    compressed.append(bundled)
                    prev_pos = idx

            compressed_sequences.append(np.array(compressed, dtype=np.int64))

        # Concatenate all beats
        flattened_sequence = np.concatenate(compressed_sequences)

        return flattened_sequence

    def compress_tokens_velocity(
        self,
        token_matrix: np.ndarray,
        track_marker_id: int
    ) -> np.ndarray:
        """Compress a velocity-aware token sequence using absolute-position encoding.

        Args:
            token_matrix: dual-channel token matrix of shape
                (2, num_beats, beats_length); ch0 = pattern tokens,
                ch1 = velocity tokens (0-num_velocities).
            track_marker_id: track marker ID.

        Returns:
            compressed_sequence: compressed 1-D sequence.

        Encoding format:
            non-empty: [track_marker] [abs_pos0] [token0] [velocity0+offset] [abs_pos1] [token1] [velocity1+offset] ...
                       - each note uses an absolute-position token.
                       - velocity values are shifted by velocity_offset.
            empty:     [track_marker, 0]
        """
        assert token_matrix.shape[0] == 2, f"Expected 2 channels, got {token_matrix.shape[0]}"

        pattern_matrix = token_matrix[0]   # shape: (num_beats, beats_length)
        velocity_matrix = token_matrix[1]  # shape: (num_beats, beats_length)

        compressed_sequences = []

        for beat_idx in range(pattern_matrix.shape[0]):
            beat_tokens = pattern_matrix[beat_idx]
            beat_velocities = velocity_matrix[beat_idx]

            # Find the positions of all nonzero tokens
            non_zero_indices = np.where(beat_tokens != 0)[0]

            if len(non_zero_indices) == 0:
                # Empty beat
                compressed = [track_marker_id, 0]
            else:
                # Non-empty beat - start with the marker
                compressed = [track_marker_id]

                # Append [absolute position, token value, velocity value] triples
                for idx in non_zero_indices:
                    position_marker = self.pattern_num + idx  # absolute position
                    token_value = beat_tokens[idx]
                    velocity_value = beat_velocities[idx] + self.velocity_offset

                    compressed.extend([position_marker, token_value, velocity_value])

            compressed_sequences.append(np.array(compressed, dtype=np.int64))

        # Concatenate all beats
        flattened_sequence = np.concatenate(compressed_sequences)

        return flattened_sequence

    # ==================== Decoding methods ====================

    def decompress_tokens(
        self,
        compressed_sequence: Union[np.ndarray, list],
        split_marker_id: int,
        empty_marker_id: int
    ) -> np.ndarray:
        """Decompress a bundled token sequence -> token matrix.

        Args:
            compressed_sequence: compressed token sequence.
            split_marker_id: voice split-marker ID.
            empty_marker_id: empty-segment marker ID.

        Returns:
            decompressed_beats: token matrix of shape (num_beats, beats_length).
        """
        if isinstance(compressed_sequence, list):
            compressed_sequence = np.array(compressed_sequence, dtype=np.int64)

        decompressed_beats = []
        i = 0

        while i < len(compressed_sequence):
            current_token = compressed_sequence[i]

            if current_token == empty_marker_id:
                # Empty beat
                decompressed_beats.append(np.zeros(self.beats_length, dtype=np.int64))
                i += 1

            elif current_token == split_marker_id:
                # Non-empty beat
                i += 1  # skip split_marker
                beat = np.zeros(self.beats_length, dtype=np.int64)
                current_pos = 0

                # Read bundled tokens until a special marker (>= empty_marker_id) is reached
                while i < len(compressed_sequence):
                    next_token = compressed_sequence[i]

                    # A special marker ends the current beat
                    if next_token >= empty_marker_id:
                        break

                    # Decode the bundled token
                    relative_pos = int(next_token) // self.pattern_num
                    token_value = int(next_token) % self.pattern_num

                    abs_pos = current_pos + relative_pos
                    if 0 <= abs_pos < self.beats_length:
                        beat[abs_pos] = token_value
                    current_pos = abs_pos

                    i += 1

                decompressed_beats.append(beat)

            else:
                i += 1  # skip unknown token

        if len(decompressed_beats) == 0:
            return np.zeros((1, self.beats_length), dtype=np.int64)

        return np.stack(decompressed_beats, axis=0)

    def decompress_tokens_velocity(
        self,
        compressed_sequence: Union[np.ndarray, list],
        track_marker_id: int
    ) -> np.ndarray:
        """Decompress a velocity-aware token sequence into a dual-channel token matrix.

        Args:
            compressed_sequence: compressed token sequence, formatted as
                [track_marker] [abs_pos0] [token0] [velocity0+offset] [abs_pos1] [token1] [velocity1+offset] ...
            track_marker_id: track marker ID.

        Returns:
            decompressed_beats: dual-channel token matrix of shape
                (2, num_beats, beats_length); ch0 = pattern tokens,
                ch1 = velocity tokens (0-num_velocities) with velocity_offset
                already subtracted.
        """
        if isinstance(compressed_sequence, list):
            compressed_sequence = np.array(compressed_sequence, dtype=np.int64)

        decompressed_pattern_beats = []
        decompressed_velocity_beats = []
        i = 0

        while i < len(compressed_sequence):
            current_token = compressed_sequence[i]
            if current_token == track_marker_id:
                # Non-empty beat - the current token is the start marker
                i += 1  # skip start_marker
                pattern_beat = np.zeros(self.beats_length, dtype=np.int64)
                velocity_beat = np.zeros(self.beats_length, dtype=np.int64)

                # Read position-token-velocity triples until the next beat starts
                while i < len(compressed_sequence):
                    position_marker = compressed_sequence[i]

                    if position_marker == 0:
                        # End marker of an empty beat
                        i += 1
                        break

                    # Check whether this is the start of the next beat
                    if position_marker == track_marker_id:
                        break

                    i += 1

                    if i >= len(compressed_sequence):
                        break

                    # Read the token value
                    token_value = compressed_sequence[i]
                    i += 1

                    if i >= len(compressed_sequence):
                        break

                    # Read the velocity value
                    velocity_value = compressed_sequence[i]
                    i += 1

                    # Compute the absolute position directly
                    abs_pos = position_marker - self.pattern_num

                    # Fill in token and velocity (subtract offset to restore the raw velocity token)
                    if 0 <= abs_pos < self.beats_length:
                        pattern_beat[abs_pos] = token_value
                        velocity_beat[abs_pos] = velocity_value - self.velocity_offset

                decompressed_pattern_beats.append(pattern_beat)
                decompressed_velocity_beats.append(velocity_beat)

        pattern_sequence = np.stack(decompressed_pattern_beats, axis=0)
        velocity_sequence = np.stack(decompressed_velocity_beats, axis=0)

        # Stack into dual-channel output: (2, num_beats, beats_length)
        decompressed_sequence = np.stack([pattern_sequence, velocity_sequence], axis=0)

        return decompressed_sequence

    def patch_tokens_to_image(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """Reconstruct a dual-channel piano roll from tokens.

        Args:
            tokens: token matrix of shape (num_time_patches, num_pitch_patches).

        Returns:
            image: dual-channel piano roll of shape (2, 88, t);
                ch0 = sustain, ch1 = onset.
        """
        num_patch_cols, num_patch_rows = tokens.shape

        # Decode tokens into their base-3 representation
        combined_patches = np.zeros(
            (num_patch_cols, num_patch_rows, self.patch_size),
            dtype=np.int64
        )
        temp_tokens = tokens.copy()

        for i in range(self.patch_size):
            combined_patches[:, :, i] = temp_tokens // self.powers_3[i]
            temp_tokens = temp_tokens % self.powers_3[i]

        # Recover the two channels from the ternary values
        # 0 -> sustain=0, onset=0
        # 1 -> onset (sustain=1, onset=1)
        # 2 -> sustain continuation (sustain=1, onset=0)
        sustain_patches = (combined_patches >= 1).astype(np.float32)
        onset_patches = (combined_patches == 1).astype(np.float32)

        # Rebuild the image
        sustain_channel = self._patches_to_channel(
            sustain_patches,
            num_patch_cols,
            num_patch_rows
        )
        onset_channel = self._patches_to_channel(
            onset_patches,
            num_patch_cols,
            num_patch_rows
        )

        # Stack into two channels
        image = np.stack([sustain_channel, onset_channel], axis=0)
        return image

    def patch_tokens_to_image_velocity(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """Reconstruct a dual-channel piano roll from velocity-aware tokens.

        Args:
            tokens: dual-channel token matrix of shape
                (2, num_time_patches, num_pitch_patches);
                ch0 = ternary-encoded note pattern,
                ch1 = quantized velocity (0-num_velocities).

        Returns:
            image: dual-channel piano roll of shape (2, 88, t);
                ch0 = sustain (velocity), ch1 = onset (velocity);
                velocity values are 0-127 (dequantized).
        """
        assert tokens.shape[0] == 2, f"Expected 2 channels, got {tokens.shape[0]}"

        pattern_tokens = tokens[0]  # shape: (num_time_patches, num_pitch_patches)
        velocity_tokens = tokens[1]  # shape: (num_time_patches, num_pitch_patches)

        num_patch_cols, num_patch_rows = pattern_tokens.shape

        # Decode pattern tokens into their base-3 representation
        combined_patches = np.zeros(
            (num_patch_cols, num_patch_rows, self.patch_size),
            dtype=np.int64
        )
        temp_tokens = pattern_tokens.copy()

        for i in range(self.patch_size):
            combined_patches[:, :, i] = temp_tokens // self.powers_3[i]
            temp_tokens = temp_tokens % self.powers_3[i]

        # Recover the binary masks of both channels from the ternary values
        # 0 -> sustain=0, onset=0
        # 1 -> onset (sustain=1, onset=1)
        # 2 -> sustain continuation (sustain=1, onset=0)
        sustain_mask = (combined_patches >= 1).astype(np.float32)
        onset_mask = (combined_patches == 1).astype(np.float32)

        # Dequantize velocity: restore from 1-num_velocities back to 1-127
        dequantized_velocity = self.dequantize_velocity(velocity_tokens)

        # Apply the velocity to each patch
        # dequantized_velocity shape: (num_patch_cols, num_patch_rows)
        # Expand to (num_patch_cols, num_patch_rows, patch_size)
        velocity_expanded = np.repeat(
            dequantized_velocity[:, :, np.newaxis],
            self.patch_size,
            axis=2
        ).astype(np.float32)

        # Apply velocity to the sustain and onset masks
        sustain_patches = sustain_mask * velocity_expanded
        onset_patches = onset_mask * velocity_expanded

        # Rebuild the image
        sustain_channel = self._patches_to_channel(
            sustain_patches,
            num_patch_cols,
            num_patch_rows
        )
        onset_channel = self._patches_to_channel(
            onset_patches,
            num_patch_cols,
            num_patch_rows
        )

        # Stack into two channels
        image = np.stack([sustain_channel, onset_channel], axis=0)
        return image

    def _patches_to_channel(
        self,
        patches: np.ndarray,
        num_patch_cols: int,
        num_patch_rows: int
    ) -> np.ndarray:
        """Rebuild a channel from patches."""
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_h, self.patch_w)
        patches = patches.transpose(1, 2, 0, 3)  # (rows, h, cols, w)
        channel = patches.reshape(self.beats_length, num_patch_cols * self.patch_w)
        return channel

    # ==================== Utility methods ====================

    def get_config(self) -> dict:
        """Return the configuration as a dict."""
        return {
            'patch_h': self.patch_h,
            'patch_w': self.patch_w,
            'pattern_num': self.pattern_num,
            'beats_length': self.beats_length,
        }

    def __repr__(self) -> str:
        return (
            f"PianoRollTokenizer("
            f"patch_size={self.patch_h}x{self.patch_w}, "
            f"beats_length={self.beats_length}, "
            f"pattern_num={self.pattern_num})"
        )


