"""
Token to MIDI Converter for FELIX-Music.

Converts model output token sequences to MIDI files.
Adapted from encoding/with_pair/token2midi.py with local imports.
"""

import numpy as np
import torch
import pretty_midi
from typing import Union, Optional, Tuple, List

from my_tokenizer import PianoRollTokenizer
from config import MusicTokenConfig

_tc = MusicTokenConfig()
BAR_TOKEN = _tc.bar_token_id
BOS_TOKEN = _tc.bos_token_id
EOS_TOKEN = _tc.eos_token_id
SPLIT_0 = _tc.split_0_id
SPLIT_1 = _tc.split_1_id
EMPTY_MARKER = _tc.empty_marker_id
TIME_SIG_OFFSET = _tc.time_sig_offset_id
BPM_OFFSET = _tc.bpm_offset_id


def create_tokenizer():
    """Create a PianoRollTokenizer with the Scheme D (absolute bundled) config."""
    return PianoRollTokenizer(
        patch_h=_tc.patch_h,
        patch_w=_tc.patch_w,
        pattern_num=_tc.pattern_num,
        beats_length=_tc.beats_length,
    )


class Token2MIDI:
    """
    Token sequence to MIDI converter.

    Flow:
    1. Extract metadata (time signature, BPM)
    2. Split by bars
    3. For each bar: separate tracks, decode to pianoroll
    4. Concatenate all bars
    5. Convert to MIDI
    """

    def __init__(self, tokenizer: Optional[PianoRollTokenizer] = None, patch_w: int = 4):
        if tokenizer is None:
            tokenizer = create_tokenizer()
        self.tokenizer = tokenizer
        self.patch_w = patch_w

    def convert(
        self,
        token_sequence: Union[torch.Tensor, np.ndarray, list],
        output_path: str,
        tempo: Optional[int] = None,
        velocity: int = 64,
        merge_tracks: bool = False,
    ) -> str:
        """Convert token sequence to MIDI file."""
        if isinstance(token_sequence, torch.Tensor):
            token_sequence = token_sequence.cpu().numpy()
        elif isinstance(token_sequence, list):
            token_sequence = np.array(token_sequence)

        # Extract metadata
        time_signature, bpm = self._extract_metadata(token_sequence)
        if tempo is None:
            tempo = bpm

        # Extract music content
        music_tokens = self._extract_music_tokens(token_sequence)

        # Split by bars
        bars = self._split_by_bars(music_tokens)

        # Process each bar
        track0_pianorolls = []
        track1_pianorolls = []

        for bar_tokens in bars:
            track0_pr, track1_pr = self._process_bar(bar_tokens)
            track0_pr, track1_pr = self._align_tracks(track0_pr, track1_pr)
            track0_pianorolls.append(track0_pr)
            track1_pianorolls.append(track1_pr)

        if len(track0_pianorolls) == 0:
            # Empty sequence
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            midi.write(output_path)
            return output_path

        track0_full = np.concatenate(track0_pianorolls, axis=-1)
        track1_full = np.concatenate(track1_pianorolls, axis=-1)

        combined_pianoroll = np.concatenate([track0_full, track1_full], axis=0)
        combined_pianoroll = combined_pianoroll[:, ::-1, :].copy()

        if merge_tracks:
            combined_pianoroll = np.maximum(track0_full, track1_full)
            combined_pianoroll = combined_pianoroll[:, ::-1, :].copy()
            self._pianoroll_to_midi_single_track(combined_pianoroll, output_path, tempo, velocity)
        else:
            self._pianoroll_to_midi(combined_pianoroll, output_path, tempo, velocity)

        return output_path

    def _extract_metadata(self, token_sequence: np.ndarray) -> Tuple[str, int]:
        """Extract time signature and BPM from header."""
        time_sig_map = {0: '4/4', 1: '3/4', 2: '2/4', 3: '6/8', 4: '2/2'}
        bpm_map = {0: 80, 1: 120, 2: 220, 3: 120}

        time_sig_token = int(token_sequence[1])
        time_sig_idx = time_sig_token - TIME_SIG_OFFSET
        time_signature = time_sig_map.get(time_sig_idx, '4/4')

        bpm_token = int(token_sequence[2])
        bpm_idx = bpm_token - BPM_OFFSET
        bpm = bpm_map.get(bpm_idx, 120)

        return time_signature, bpm

    def _extract_music_tokens(self, token_sequence: np.ndarray) -> np.ndarray:
        """Extract music content (remove BOS, metadata, EOS)."""
        start_idx = 3
        end_idx = len(token_sequence)
        if token_sequence[-1] == EOS_TOKEN:
            end_idx = -1
        return token_sequence[start_idx:end_idx]

    def _split_by_bars(self, music_tokens: np.ndarray) -> List[np.ndarray]:
        """Split by BAR_TOKEN."""
        bar_positions = np.where(music_tokens == BAR_TOKEN)[0]
        bars = []
        for i in range(len(bar_positions)):
            start = bar_positions[i] + 1
            end = bar_positions[i + 1] if i + 1 < len(bar_positions) else len(music_tokens)
            bar_tokens = music_tokens[start:end]
            if len(bar_tokens) > 0:
                bars.append(bar_tokens)
        return bars

    def _process_bar(self, bar_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single bar: separate tracks and decode."""
        track0_beats, track1_beats = self._separate_tracks(bar_tokens)
        track0_pr = self._decode_track_beats(track0_beats, SPLIT_0)
        track1_pr = self._decode_track_beats(track1_beats, SPLIT_1)
        return track0_pr, track1_pr

    def _separate_tracks(self, bar_tokens: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Separate interleaved tracks: even=T0, odd=T1."""
        segments = []
        i = 0
        while i < len(bar_tokens):
            token = bar_tokens[i]
            if token == SPLIT_0 or token == SPLIT_1:
                segment_start = i
                i += 1
                while i < len(bar_tokens) and bar_tokens[i] < EMPTY_MARKER:
                    i += 1
                segments.append(bar_tokens[segment_start:i])
            elif token == EMPTY_MARKER:
                segments.append(bar_tokens[i:i + 1])
                i += 1
            else:
                i += 1

        track0_beats = []
        track1_beats = []
        for idx, seg in enumerate(segments):
            if idx % 2 == 0:
                track0_beats.append(seg)
            else:
                track1_beats.append(seg)

        return track0_beats, track1_beats

    def _decode_track_beats(self, beat_token_list: List[np.ndarray], split_id: int) -> np.ndarray:
        """Decode a track's beats to pianoroll."""
        beat_pianorolls = []
        for beat_tokens in beat_token_list:
            if len(beat_tokens) == 1 and beat_tokens[0] == EMPTY_MARKER:
                pianoroll = np.zeros((2, 88, self.patch_w), dtype=np.float32)
            else:
                token_matrix = self.tokenizer.decompress_tokens(
                    beat_tokens,
                    split_marker_id=split_id,
                    empty_marker_id=EMPTY_MARKER,
                )
                pianoroll = self.tokenizer.patch_tokens_to_image(token_matrix)
            beat_pianorolls.append(pianoroll)

        if len(beat_pianorolls) > 0:
            return np.concatenate(beat_pianorolls, axis=-1)
        return np.zeros((2, 88, self.patch_w), dtype=np.float32)

    def _align_tracks(self, track0: np.ndarray, track1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two tracks to the same time length."""
        t0, t1 = track0.shape[-1], track1.shape[-1]
        if t0 == t1:
            return track0, track1
        max_len = max(t0, t1)
        if t0 < max_len:
            track0 = np.pad(track0, ((0, 0), (0, 0), (0, max_len - t0)), mode='constant')
        if t1 < max_len:
            track1 = np.pad(track1, ((0, 0), (0, 0), (0, max_len - t1)), mode='constant')
        return track0, track1

    def _pianoroll_to_midi(self, pianoroll: np.ndarray, output_path: str, tempo: int, velocity: int):
        """Convert 4-channel pianoroll to MIDI."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        seconds_per_step = 60.0 / tempo / 4

        track0_instrument = self._create_midi_track(pianoroll[0], pianoroll[1], seconds_per_step, velocity)
        midi.instruments.append(track0_instrument)

        track1_instrument = self._create_midi_track(pianoroll[2], pianoroll[3], seconds_per_step, velocity)
        midi.instruments.append(track1_instrument)

        midi.write(output_path)

    def _pianoroll_to_midi_single_track(self, pianoroll: np.ndarray, output_path: str, tempo: int, velocity: int):
        """Convert 2-channel pianoroll to single-track MIDI."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        seconds_per_step = 60.0 / tempo / 4
        instrument = self._create_midi_track(pianoroll[0], pianoroll[1], seconds_per_step, velocity)
        midi.instruments.append(instrument)
        midi.write(output_path)

    def _create_midi_track(
        self, sustain_roll: np.ndarray, onset_roll: np.ndarray,
        seconds_per_step: float, velocity: int, program: int = 0
    ) -> pretty_midi.Instrument:
        """Create MIDI instrument from sustain and onset pianorolls."""
        instrument = pretty_midi.Instrument(program=program)
        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]
            for onset_pos in onset_positions:
                end_pos = onset_pos + 1
                while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                    end_pos += 1
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=onset_pos * seconds_per_step,
                    end=end_pos * seconds_per_step,
                )
                instrument.notes.append(note)
        return instrument
