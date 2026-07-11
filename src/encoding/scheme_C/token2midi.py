"""
Token to MIDI Converter

Full pipeline to convert a model-generated token sequence into a MIDI file:
1. Extract metadata (time signature, BPM).
2. Split into bars by bar_token_id.
3. For each bar, separate the two tracks and decode them into piano rolls.
4. Concatenate all bars.
5. Convert to a MIDI file.

Author: Refactored version
"""

import numpy as np
import torch
import pretty_midi
from typing import Union, Optional, Tuple, List
from my_tokenizer import PianoRollTokenizer


class Token2MIDI:
    """Converter from a token sequence to MIDI.

    Pipeline:
    1. Extract metadata (time signature, BPM) from the token sequence.
    2. Split into bars by bar_token_id.
    3. For each bar:
       - Separate the beat tokens of track0 and track1.
       - Decode into piano rolls using the tokenizer.
       - Concatenate into a full bar.
    4. Merge all bars.
    5. Align the time length of the two tracks.
    6. Convert to a MIDI file.

    Args:
        tokenizer: PianoRollTokenizer instance.
        config: ModelConfig object.

    Example:
        >>> from my_tokenizer import PianoRollTokenizer
        >>> from config import ModelConfig
        >>>
        >>> config = ModelConfig()
        >>> tokenizer = PianoRollTokenizer(
        ...     patch_h=config.patch_h,
        ...     patch_w=config.patch_w,
        ...     pattern_num=config.pattern_num,
        ...     beats_length=config.beats_length
        ... )
        >>> converter = Token2MIDI(tokenizer, config)
        >>> converter.convert(token_sequence, "output.mid", tempo=120)
    """

    def __init__(self, tokenizer: PianoRollTokenizer, config):
        """Initialize the converter.

        Args:
            tokenizer: PianoRollTokenizer instance.
            config: configuration object with the various token ID definitions.
        """
        self.tokenizer = tokenizer

        # Token ID configuration
        self.bar_token_id = config.bar_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.split_0_id = config.split_0_id
        self.split_1_id = config.split_1_id
        self.empty_marker_id = config.empty_marker_id

        # Metadata configuration
        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id

        # Patch configuration
        self.patch_w = config.patch_w  # number of time steps per beat

    def convert(
        self,
        token_sequence: Union[torch.Tensor, np.ndarray, list],
        output_path: str,
        tempo: Optional[int] = None,
        velocity: int = 64,
        merge_tracks: bool = False
    ) -> str:
        """Convert a token sequence into a MIDI file (main entry point).

        Args:
            token_sequence: full token sequence (including BOS/EOS).
            output_path: path to save the MIDI file.
            tempo: tempo (BPM); if None, extracted from the sequence.
            velocity: MIDI note velocity (1-127).

        Returns:
            output_path: path of the saved MIDI file.
        """
        # Convert to a numpy array
        if isinstance(token_sequence, torch.Tensor):
            token_sequence = token_sequence.cpu().numpy()
        elif isinstance(token_sequence, list):
            token_sequence = np.array(token_sequence)

        print(f"Raw token sequence length: {len(token_sequence)}")

        # 1. Extract metadata
        time_signature, bpm = self._extract_metadata(token_sequence)
        if tempo is None:
            tempo = bpm
        print(f"Metadata - time signature: {time_signature}, BPM: {bpm} (using tempo={tempo})")

        # 2. Extract the musical content (drop BOS, metadata, EOS)
        music_tokens = self._extract_music_tokens(token_sequence)
        print(f"Musical content token count: {len(music_tokens)}")

        # 3. Split into bars
        bars = self._split_by_bars(music_tokens)
        print(f"Split into {len(bars)} bars")

        # 4. Process each bar to obtain the two-track piano rolls
        track0_pianorolls = []
        track1_pianorolls = []

        for bar_idx, bar_tokens in enumerate(bars):
            track0_pr, track1_pr = self._process_bar(bar_tokens)
            # Align the two-track length within each bar
            track0_pr, track1_pr = self._align_tracks(track0_pr, track1_pr)
            track0_pianorolls.append(track0_pr)
            track1_pianorolls.append(track1_pr)

        # 5. Concatenate all bars (already aligned at bar level, no need to realign)
        track0_full = np.concatenate(track0_pianorolls, axis=-1)  # (2, 88, total_time)
        track1_full = np.concatenate(track1_pianorolls, axis=-1)
        # 6. Merge the two tracks into a 4-channel piano roll
        combined_pianoroll = np.concatenate([track0_full, track1_full], axis=0)  # (4, 88, time)
        combined_pianoroll = combined_pianoroll[:, ::-1, :].copy()
        if merge_tracks:
            # Merge into a single track if requested
            combined_pianoroll = np.maximum(track0_full, track1_full)
            self._pianoroll_to_midi_single_track(combined_pianoroll, output_path, tempo, velocity)
        else:
        # 8. Convert to MIDI
            self._pianoroll_to_midi(combined_pianoroll, output_path, tempo, velocity)

        return output_path

    def _extract_metadata(self, token_sequence: np.ndarray) -> Tuple[str, int]:
        """Extract the time signature and BPM.

        Sequence format: [BOS, time_sig, bpm, music_content..., EOS].

        Args:
            token_sequence: full token sequence.

        Returns:
            time_signature: time-signature string (e.g. "4/4").
            bpm: tempo.
        """
        # Time-signature map (matches time_sig_offset_id in config.py)
        time_sig_map = {
            0: '4/4',
            1: '3/4',
            2: '2/4',
            3: '6/8',
            4: '2/2'
        }

        # BPM map (matches the encode_bpm function in PianoDataset.py)
        bpm_map = {
            0: 80,   # slow <90
            1: 120,  # medium 90-200
            2: 220,  # fast >200
            3: 120   # unknown, default 120
        }

        # Extract the time signature (position 1)
        time_sig_token = int(token_sequence[1])
        time_sig_idx = time_sig_token - self.time_sig_offset_id
        time_signature = time_sig_map.get(time_sig_idx, '4/4')

        # Extract the BPM (position 2)
        bpm_token = int(token_sequence[2])
        bpm_idx = bpm_token - self.bpm_offset_id
        bpm = bpm_map.get(bpm_idx, 120)

        return time_signature, bpm

    def _extract_music_tokens(self, token_sequence: np.ndarray) -> np.ndarray:
        """Extract the musical content tokens (drop BOS, metadata, EOS).

        Args:
            token_sequence: full token sequence.

        Returns:
            music_tokens: token sequence of pure musical content.
        """
        # Drop BOS (position 0), time signature (position 1), BPM (position 2)
        start_idx = 3

        # Find and drop EOS
        end_idx = len(token_sequence)
        if token_sequence[-1] == self.eos_token_id:
            end_idx = -1

        return token_sequence[start_idx:end_idx]

    def _split_by_bars(self, music_tokens: np.ndarray) -> List[np.ndarray]:
        """Split into bars by bar_token_id.

        Args:
            music_tokens: token sequence of musical content.

        Returns:
            bars: list of bar tokens; each element is one bar's tokens
                (excluding bar_token_id itself).
        """
        # Find the positions of all bar tokens
        bar_positions = np.where(music_tokens == self.bar_token_id)[0]

        bars = []
        for i in range(len(bar_positions)):
            start = bar_positions[i] + 1  # skip the bar token itself
            end = bar_positions[i + 1] if i + 1 < len(bar_positions) else len(music_tokens)
            bar_tokens = music_tokens[start:end]
            if len(bar_tokens) > 0:  # ignore empty bars
                bars.append(bar_tokens)

        return bars

    def _process_bar(self, bar_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single bar: separate the two tracks and decode into piano rolls.

        In-bar format (bundled encoding, interleaved):
        [upper-voice beat0] [lower-voice beat0] [upper-voice beat1] [lower-voice beat1] ...

        Args:
            bar_tokens: token sequence of a single bar.

        Returns:
            track0_pianoroll: piano roll of track0 (2, 88, time).
            track1_pianoroll: piano roll of track1 (2, 88, time).
        """
        # Separate the beats of track0 and track1
        track0_beats, track1_beats = self._separate_tracks(bar_tokens)

        # Decode each track
        track0_pianoroll = self._decode_track_beats(track0_beats, self.split_0_id)
        track1_pianoroll = self._decode_track_beats(track1_beats, self.split_1_id)

        return track0_pianoroll, track1_pianoroll

    def _separate_tracks(self, bar_tokens: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Separate the beat segments of track0 and track1 within a bar (bundled encoding).

        Parses segments delimited by SPLIT_0/SPLIT_1/EMPTY_MARKER in order,
        then assigns them by interleave order: even index -> track0, odd
        index -> track1.

        Args:
            bar_tokens: all tokens within a bar.

        Returns:
            track0_beats: list of track0 beat tokens.
            track1_beats: list of track1 beat tokens.
        """
        segments = []
        i = 0

        while i < len(bar_tokens):
            token = bar_tokens[i]

            if token == self.split_0_id or token == self.split_1_id:
                # Non-empty segment: starts at the split marker, read until the next special token
                segment_start = i
                i += 1
                while i < len(bar_tokens) and bar_tokens[i] < self.empty_marker_id:
                    i += 1
                segments.append(bar_tokens[segment_start:i])

            elif token == self.empty_marker_id:
                # Empty segment
                segments.append(bar_tokens[i:i+1])
                i += 1

            else:
                i += 1  # skip unknown token

        # Interleaved assignment: even index -> track0, odd index -> track1
        track0_beats = []
        track1_beats = []
        for idx, seg in enumerate(segments):
            if idx % 2 == 0:
                track0_beats.append(seg)
            else:
                track1_beats.append(seg)

        return track0_beats, track1_beats

    def _decode_track_beats(self, beat_token_list: List[np.ndarray], split_id: int) -> np.ndarray:
        """Decode all beats of a single track into a piano roll (bundled encoding).

        Args:
            beat_token_list: list of beat tokens.
            split_id: split-marker ID for this track.

        Returns:
            pianoroll: decoded piano roll (2, 88, total_time).
        """
        beat_pianorolls = []

        for beat_tokens in beat_token_list:
            if len(beat_tokens) == 1 and beat_tokens[0] == self.empty_marker_id:
                # Empty beat
                pianoroll = np.zeros((2, 88, self.patch_w), dtype=np.float32)
            else:
                # Decompress the bundled tokens into a token matrix
                token_matrix = self.tokenizer.decompress_tokens(
                    beat_tokens,
                    split_marker_id=split_id,
                    empty_marker_id=self.empty_marker_id
                )
                # Convert into a piano roll
                pianoroll = self.tokenizer.patch_tokens_to_image(token_matrix)  # (2, 88, patch_w)

            beat_pianorolls.append(pianoroll)

        # Concatenate all beats
        if len(beat_pianorolls) > 0:
            full_pianoroll = np.concatenate(beat_pianorolls, axis=-1)
        else:
            # Empty track
            full_pianoroll = np.zeros((2, 88, self.patch_w), dtype=np.float32)

        return full_pianoroll

    def _align_tracks(
        self,
        track0: np.ndarray,
        track1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align the time length of two tracks (pad to the same length).

        Args:
            track0: piano roll of track0 (2, 88, t0).
            track1: piano roll of track1 (2, 88, t1).

        Returns:
            track0_aligned: aligned track0.
            track1_aligned: aligned track1.
        """
        t0 = track0.shape[-1]
        t1 = track1.shape[-1]

        if t0 == t1:
            return track0, track1

        max_len = max(t0, t1)

        # Pad the shorter track
        if t0 < max_len:
            pad_width = ((0, 0), (0, 0), (0, max_len - t0))
            track0 = np.pad(track0, pad_width, mode='constant', constant_values=0)

        if t1 < max_len:
            pad_width = ((0, 0), (0, 0), (0, max_len - t1))
            track1 = np.pad(track1, pad_width, mode='constant', constant_values=0)

        return track0, track1

    def _pianoroll_to_midi(
        self,
        pianoroll: np.ndarray,
        output_path: str,
        tempo: int,
        velocity: int
    ):
        """Convert a 4-channel piano roll into a MIDI file.

        Args:
            pianoroll: (4, 88, time) - [track0_sustain, track0_onset, track1_sustain, track1_onset].
            output_path: path to save the MIDI file.
            tempo: tempo (BPM).
            velocity: MIDI note velocity.
        """
        # Create the MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Compute the seconds per time step (based on patch_w, usually a 1/16 note)
        seconds_per_step = 60.0 / tempo / 4  # assume patch_w=4 maps to one beat, each step a 1/16 beat

        # Process track0 (usually the upper voice / melody)
        track0_sustain = pianoroll[0]  # (88, time)
        track0_onset = pianoroll[1]
        piano_track0 = self._create_midi_track(
            track0_sustain,
            track0_onset,
            seconds_per_step,
            velocity,
            program=0  # Acoustic Grand Piano
        )
        midi.instruments.append(piano_track0)

        # Process track1 (usually the lower voice / accompaniment)
        track1_sustain = pianoroll[2]
        track1_onset = pianoroll[3]
        piano_track1 = self._create_midi_track(
            track1_sustain,
            track1_onset,
            seconds_per_step,
            velocity,
            program=0
        )
        midi.instruments.append(piano_track1)

        # Save the MIDI file
        midi.write(output_path)

        print(f"\nMIDI file saved to: {output_path}")
        print(f"Total duration: {midi.get_end_time():.2f} s")
        print(f"Track0 note count: {len(piano_track0.notes)}")
        print(f"Track1 note count: {len(piano_track1.notes)}")

    def _pianoroll_to_midi_single_track(
        self,
        pianoroll: np.ndarray,
        output_path: str,
        tempo: int,
        velocity: int
    ):
        """Convert a 4-channel piano roll into a MIDI file.

        Args:
            pianoroll: (4, 88, time) - [track0_sustain, track0_onset, track1_sustain, track1_onset].
            output_path: path to save the MIDI file.
            tempo: tempo (BPM).
            velocity: MIDI note velocity.
        """
        # Create the MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Compute the seconds per time step (based on patch_w, usually a 1/16 note)
        seconds_per_step = 60.0 / tempo / 4  # assume patch_w=4 maps to one beat, each step a 1/16 beat

        # Process track0 (usually the upper voice / melody)
        track0_sustain = pianoroll[0]  # (88, time)
        track0_onset = pianoroll[1]
        piano_track0 = self._create_midi_track(
            track0_sustain,
            track0_onset,
            seconds_per_step,
            velocity,
            program=0  # Acoustic Grand Piano
        )
        midi.instruments.append(piano_track0)


        # Save the MIDI file
        midi.write(output_path)

        print(f"\nMIDI file saved to: {output_path}")
        print(f"Total duration: {midi.get_end_time():.2f} s")
        print(f"Track0 note count: {len(piano_track0.notes)}")

    def _create_midi_track(
        self,
        sustain_roll: np.ndarray,
        onset_roll: np.ndarray,
        seconds_per_step: float,
        velocity: int,
        program: int = 0
    ) -> pretty_midi.Instrument:
        """Create a MIDI track from sustain and onset piano rolls.

        Args:
            sustain_roll: (88, time) sustain channel.
            onset_roll: (88, time) onset channel.
            seconds_per_step: seconds per time step.
            velocity: note velocity.
            program: MIDI program number.

        Returns:
            instrument: PrettyMIDI instrument object.
        """
        instrument = pretty_midi.Instrument(program=program)

        # Iterate over each pitch
        for pitch_idx in range(88):
            pitch = pitch_idx + 21  # MIDI pitch (A0=21 and up)

            # Find all onset positions
            onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]

            for onset_pos in onset_positions:
                # Find the note end position (end of sustain)
                end_pos = onset_pos + 1

                # Keep searching forward until sustain ends
                while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                    end_pos += 1

                # Convert to time (seconds)
                start_time = onset_pos * seconds_per_step
                end_time = end_pos * seconds_per_step

                # Create the MIDI note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

        return instrument
