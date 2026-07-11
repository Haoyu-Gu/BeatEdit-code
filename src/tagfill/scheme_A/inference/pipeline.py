"""
FELIX-Music Inference Pipeline.

End-to-end: input token sequence (melody + optional old accompaniment)
→ Tagger (per-token labels) → skeleton → Inserter (fill MASKs) → final output.

Supports:
- Accompaniment re-editing (old accomp → mixed labels)
- Accompaniment generation from scratch (empty accomp → APPEND labels)
- Iterative refinement (multiple Tagger→Inserter passes)
- Temperature / top-k sampling for Inserter
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from collections import Counter

from configs.config import (
    LABEL_KEEP, LABEL_REPLACE, MASK_TOKEN, PAD_TOKEN, BAR_TOKEN,
    TaggerConfig, InserterConfig, FELIXTrainingConfig,
)
from data.sequence_parser import (
    parse_sequence, separate_tracks, rebuild_interleaved,
    decode_beat, encode_beat,
)
from data.skeleton_builder import build_skeleton
from models.tagger import FELIXTagger
from models.inserter import FELIXInserter


class FELIXPipeline:
    """
    End-to-end FELIX inference pipeline.

    Usage:
        pipeline = FELIXPipeline('checkpoints/tagger/tagger_best.pt',
                                  'checkpoints/inserter/inserter_best.pt')
        output = pipeline.generate(token_sequence)
        # or for generation from scratch:
        output = pipeline.generate_from_melody(token_sequence)
    """

    def __init__(self, tagger, inserter, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if isinstance(tagger, str):
            tagger = self._load_tagger(tagger)
        if isinstance(inserter, str):
            inserter = self._load_inserter(inserter)

        self.tagger = tagger.to(self.device).eval()
        self.inserter = inserter.to(self.device).eval()

    def _load_tagger(self, path):
        model = FELIXTagger(TaggerConfig())
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def _load_inserter(self, path):
        model = FELIXInserter(InserterConfig())
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    @torch.no_grad()
    def generate(
        self,
        token_sequence: List[int],
        num_iterations: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        inserter_steps: int = 2,
        confidence_threshold: float = 0.5,
        tagger_confidence_threshold: float = 0.0,
        harmony_weight: float = 0.0,
    ) -> List[int]:
        """
        Run the full FELIX pipeline.

        Args:
            token_sequence: input token sequence (melody + accompaniment).
                For generation from scratch, use generate_from_melody().
            num_iterations: number of Tagger->Inserter passes.
            temperature: sampling temperature for Inserter (1.0=no change).
            top_k: if > 0, sample from top-k tokens.
            inserter_steps: number of iterative decoding rounds for the Inserter.
                When > 1, uses confidence-guided iterative decoding (MaskGIT-style
                linear schedule): each round unmasks the highest-confidence MASK
                positions and re-runs the Inserter on the partially filled skeleton.
                Default 1 preserves the original single-pass behavior.
            confidence_threshold: minimum softmax probability for a MASK prediction
                to be accepted during iterative inserter decoding. Only used when
                inserter_steps > 1. Default 0.5.
            tagger_confidence_threshold: when > 0, tagger predictions with
                confidence below this threshold (for non-KEEP labels) are
                overridden to REPLACE, delegating the decision to the Inserter.
                Default 0.0 disables this (backward compatible).
            harmony_weight: when > 0.0, applies harmony-aware post-processing
                after inserter fills MASKs. Detects the key from melody pitch
                classes and shifts out-of-key accompaniment notes to the nearest
                in-key pitch. Default 0.0 disables this (backward compatible).

        Returns:
            Final token sequence with generated/edited accompaniment.
        """
        current_tokens = list(token_sequence)

        for iteration in range(num_iterations):
            # Step 1: Tagger -> per-token labels (with optional confidence)
            if tagger_confidence_threshold > 0.0:
                labels, confidences = self._run_tagger(
                    current_tokens, return_confidence=True
                )
                # Override low-confidence non-KEEP labels to REPLACE
                for i in range(len(labels)):
                    if labels[i] != LABEL_KEEP and confidences[i] < tagger_confidence_threshold:
                        labels[i] = LABEL_REPLACE
            else:
                labels = self._run_tagger(current_tokens)

            # Check convergence: all KEEP -> stop
            if all(l == LABEL_KEEP for l in labels):
                break

            # Step 2: Build skeleton (no targets at inference time)
            dummy_targets = [[]] * len(labels)
            skeleton_result = build_skeleton(current_tokens, labels, dummy_targets)

            skeleton_tokens = skeleton_result['skeleton_tokens']
            mask_positions = skeleton_result['mask_positions']

            if len(mask_positions) == 0:
                current_tokens = skeleton_tokens
                continue

            # Step 3: Inserter -> fill MASKs
            if inserter_steps > 1:
                current_tokens = self._run_inserter_iterative(
                    skeleton_tokens, mask_positions, temperature, top_k,
                    inserter_steps, confidence_threshold
                )
            else:
                current_tokens = self._run_inserter(
                    skeleton_tokens, mask_positions, temperature, top_k
                )

            # Step 4: Harmony-aware post-processing (optional)
            if harmony_weight > 0.0:
                current_tokens = self._apply_harmony_constraints(
                    current_tokens, harmony_weight
                )

        return current_tokens

    @torch.no_grad()
    def generate_from_melody(
        self,
        token_sequence: List[int],
        num_iterations: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        inserter_steps: int = 2,
        confidence_threshold: float = 0.5,
        tagger_confidence_threshold: float = 0.0,
        harmony_weight: float = 0.0,
    ) -> List[int]:
        """
        Generate accompaniment from scratch (blank all accompaniment beats first).

        Args:
            token_sequence: full token sequence (melody + any accompaniment).
                Accompaniment will be cleared before generation.
            inserter_steps: see generate().
            confidence_threshold: see generate().
            tagger_confidence_threshold: see generate().
            harmony_weight: see generate().
        """
        parsed = parse_sequence(token_sequence)
        melody_beats, accomp_beats = separate_tracks(parsed)

        # Clear all accompaniment to empty beats
        empty_accomp = []
        for beat in accomp_beats:
            empty_accomp.append({
                'tokens': [],
                'split_id': None,
                'start_idx': beat.get('start_idx', 0),
                'end_idx': beat.get('end_idx', 0),
            })

        empty_seq = rebuild_interleaved(melody_beats, empty_accomp, parsed)
        return self.generate(
            empty_seq, num_iterations, temperature, top_k,
            inserter_steps, confidence_threshold,
            tagger_confidence_threshold, harmony_weight
        )

    @torch.no_grad()
    def generate_inpainting(
        self,
        token_sequence: List[int],
        mask_start_beat: int,
        mask_end_beat: int,
        mask_track: int = 1,
        inserter_steps: int = 2,
        confidence_threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: int = 0,
        harmony_weight: float = 0.0,
    ) -> List[int]:
        """
        Inpainting: skip Tagger, mask specified beat range, Inserter fills.

        Args:
            token_sequence: full token sequence (melody + accompaniment).
            mask_start_beat: first beat to mask (inclusive, 0-based).
            mask_end_beat: last beat to mask (exclusive).
            mask_track: 0=melody, 1=accompaniment (default), 2=both.
            inserter_steps: MaskGIT iterative decoding rounds (1=single pass).
            confidence_threshold: min prob for iterative decoding acceptance.
            temperature: Inserter sampling temperature.
            top_k: top-k sampling (0=greedy).
            harmony_weight: post-hoc harmony correction weight.

        Returns:
            Token sequence with inpainted region filled.
        """
        parsed = parse_sequence(token_sequence)
        melody_beats, accomp_beats = separate_tracks(parsed)

        if mask_track in (1, 2):
            for i in range(mask_start_beat, min(mask_end_beat, len(accomp_beats))):
                beat = accomp_beats[i]
                if len(beat['tokens']) > 0:
                    beat['tokens'] = [MASK_TOKEN] * len(beat['tokens'])

        if mask_track in (0, 2):
            for i in range(mask_start_beat, min(mask_end_beat, len(melody_beats))):
                beat = melody_beats[i]
                if len(beat['tokens']) > 0:
                    beat['tokens'] = [MASK_TOKEN] * len(beat['tokens'])

        masked_seq = rebuild_interleaved(melody_beats, accomp_beats, parsed)
        mask_positions = [i for i, t in enumerate(masked_seq) if t == MASK_TOKEN]

        if len(mask_positions) == 0:
            return list(token_sequence)

        if inserter_steps > 1:
            result = self._run_inserter_iterative(
                masked_seq, mask_positions, temperature, top_k,
                inserter_steps, confidence_threshold
            )
        else:
            result = self._run_inserter(
                masked_seq, mask_positions, temperature, top_k
            )

        if harmony_weight > 0.0:
            result = self._apply_harmony_constraints(result, harmony_weight)

        return result

    def _run_tagger(self, tokens, return_confidence=False):
        """
        Run Tagger: tokens -> per-token label predictions.

        Args:
            tokens: input token sequence.
            return_confidence: if True, also return per-token confidence scores
                (max softmax probability). Used by confidence-aware skeleton
                building (Optimization 2).

        Returns:
            If return_confidence is False: list of int label IDs.
            If return_confidence is True: (labels, confidences) where both are
                lists of the same length. confidences[i] is the max softmax
                probability for the predicted label at position i.
        """
        max_len = self.tagger.config.max_position_embeddings
        truncated = len(tokens) > max_len
        input_tokens = tokens[:max_len] if truncated else tokens

        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        logits = self.tagger(input_ids, attention_mask)  # (1, L, 11)

        if return_confidence:
            probs = F.softmax(logits[0], dim=-1)           # (L, 11)
            labels = probs.argmax(dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

            # Pad with KEEP (confidence=1.0) for truncated tail
            if truncated:
                tail_len = len(tokens) - max_len
                labels.extend([LABEL_KEEP] * tail_len)
                confidences.extend([1.0] * tail_len)

            return labels, confidences
        else:
            labels = logits[0].argmax(dim=-1).cpu().tolist()

            # Pad with KEEP for truncated tail
            if truncated:
                labels.extend([LABEL_KEEP] * (len(tokens) - max_len))

            return labels

    def _run_inserter(self, skeleton_tokens, mask_positions, temperature, top_k):
        """Run Inserter: fill MASK positions in skeleton."""
        max_len = self.inserter.config.max_position_embeddings

        # Truncate skeleton if it exceeds model's max length
        if len(skeleton_tokens) > max_len:
            skeleton_tokens = skeleton_tokens[:max_len]
            mask_positions = [p for p in mask_positions if p < max_len]

        if len(mask_positions) == 0:
            return skeleton_tokens

        skeleton_ids = torch.tensor(skeleton_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(skeleton_ids)
        mask_pos = torch.tensor(mask_positions, dtype=torch.long).unsqueeze(0).to(self.device)

        logits = self.inserter(skeleton_ids, attention_mask, mask_pos)  # (1, M, V)
        logits = logits[0]  # (M, V)

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Sampling strategy
        if top_k > 0:
            topk_vals, topk_ids = logits.topk(top_k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            predictions = topk_ids[torch.arange(len(sampled_indices)), sampled_indices]
        else:
            predictions = logits.argmax(dim=-1)

        predictions = predictions.cpu().tolist()

        # Fill MASKs in skeleton
        result = list(skeleton_tokens)
        for pos, pred in zip(mask_positions, predictions):
            if 0 <= pos < len(result):
                result[pos] = pred

        return result

    def _run_inserter_iterative(
        self, skeleton_tokens, mask_positions, temperature, top_k,
        num_steps, confidence_threshold
    ):
        """
        Confidence-guided iterative inserter decoding (MaskGIT-style).

        Instead of filling all MASKs in one pass, splits the decoding into
        multiple rounds. Each round:
          1. Runs the inserter on the current (partially filled) skeleton.
          2. Computes softmax confidence for each remaining MASK position.
          3. Accepts the most confident predictions (linear schedule, Eq. 9).
          4. Keeps low-confidence positions as MASK for the next round.

        Each round accepts the floor(|M| / (T - t + 1)) most confident
        predictions (linear schedule, paper Eq. 9).

        Args:
            skeleton_tokens: list of int token IDs with MASK_TOKEN at positions to fill.
            mask_positions: list of int indices into skeleton_tokens that are MASK.
            temperature: sampling temperature.
            top_k: if > 0, sample from top-k tokens.
            num_steps: number of iterative decoding rounds.
            confidence_threshold: minimum softmax probability to accept a prediction.
                Predictions below this threshold remain as MASK even if scheduled
                to be unmasked in the current step.

        Returns:
            List of int token IDs with all MASKs filled.
        """
        max_len = self.inserter.config.max_position_embeddings

        # Truncate if needed
        if len(skeleton_tokens) > max_len:
            skeleton_tokens = skeleton_tokens[:max_len]
            mask_positions = [p for p in mask_positions if p < max_len]

        if len(mask_positions) == 0:
            return skeleton_tokens

        result = list(skeleton_tokens)
        remaining_masks = list(mask_positions)
        total_masks = len(remaining_masks)

        for step in range(num_steps):
            if len(remaining_masks) == 0:
                break

            # Run inserter on current skeleton state
            skeleton_ids = torch.tensor(result, dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(skeleton_ids)
            mask_pos_tensor = torch.tensor(
                remaining_masks, dtype=torch.long
            ).unsqueeze(0).to(self.device)

            logits = self.inserter(skeleton_ids, attention_mask, mask_pos_tensor)  # (1, M, V)
            logits = logits[0]  # (M, V)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Compute softmax probabilities for confidence scoring
            probs = F.softmax(logits, dim=-1)  # (M, V)

            # Get predictions and their confidence
            if top_k > 0:
                topk_vals, topk_ids = logits.topk(top_k, dim=-1)
                topk_probs = F.softmax(topk_vals, dim=-1)
                sampled_indices = torch.multinomial(topk_probs, 1).squeeze(-1)
                predictions = topk_ids[torch.arange(len(sampled_indices)), sampled_indices]
                # Confidence = probability of the sampled token in the full distribution
                pred_confidences = probs[torch.arange(len(predictions)), predictions]
            else:
                predictions = logits.argmax(dim=-1)
                pred_confidences = probs.max(dim=-1).values

            predictions = predictions.cpu().tolist()
            pred_confidences = pred_confidences.cpu().tolist()

            # Linear schedule (paper Eq. 9): accept the top
            # floor(|M| / (T - t + 1)) most confident predictions this round,
            # where |M| is the number of remaining MASK positions, T the total
            # rounds and t the 1-indexed round. The last round fills everything.
            num_to_unmask = len(remaining_masks) // (num_steps - step)
            num_to_unmask = max(1, num_to_unmask)  # Always unmask at least 1

            # Sort by confidence (descending) to pick the most confident
            indexed = list(zip(range(len(remaining_masks)), pred_confidences, predictions))
            indexed.sort(key=lambda x: x[1], reverse=True)

            new_remaining = []
            unmasked_count = 0

            for idx, conf, pred in indexed:
                pos = remaining_masks[idx]
                if unmasked_count < num_to_unmask and conf >= confidence_threshold:
                    # Accept this prediction
                    result[pos] = pred
                    unmasked_count += 1
                elif step == num_steps - 1:
                    # Last step: force-fill regardless of confidence
                    result[pos] = pred
                else:
                    # Keep as MASK for next round
                    new_remaining.append(pos)

            remaining_masks = new_remaining

        return result

    def _apply_harmony_constraints(
        self, token_sequence: List[int], harmony_weight: float
    ) -> List[int]:
        """
        Harmony-aware post-processing: shift out-of-key accompaniment notes
        to the nearest in-key pitch.

        Steps:
          1. Parse the token sequence into melody and accompaniment beats.
          2. Decode melody notes to extract pitch classes.
          3. Detect the likely key (top 7 most common pitch classes = diatonic scale).
          4. For each accompaniment note whose pitch class is not in the detected key,
             shift it to the nearest in-key pitch (by +/-1 semitone).
          5. Re-encode and rebuild the token sequence.

        Piano key index mapping:
          - 88 keys indexed 0-87, where index 0 = MIDI 108 (highest C8),
            index 87 = MIDI 21 (lowest A0).
          - MIDI note = 108 - pitch_index
          - Pitch class = MIDI note % 12

        Args:
            token_sequence: the full token sequence after inserter filling.
            harmony_weight: controls the strength of correction. When > 0,
                corrections are applied. The value is currently binary
                (any positive value enables correction), but reserved for
                future use as a continuous blending factor.

        Returns:
            Corrected token sequence with harmonically adjusted accompaniment.
        """
        parsed = parse_sequence(token_sequence)
        melody_beats, accomp_beats = separate_tracks(parsed)

        # Step 1: Collect all melody pitch classes
        melody_pitch_classes = Counter()
        for beat in melody_beats:
            if len(beat['tokens']) == 0:
                continue
            notes = decode_beat(beat['tokens'])
            for abs_pitch, val in notes:
                if 0 <= abs_pitch <= 87:
                    midi_note = 108 - abs_pitch
                    pc = midi_note % 12
                    melody_pitch_classes[pc] += 1

        if len(melody_pitch_classes) == 0:
            # No melody notes found, nothing to constrain against
            return token_sequence

        # Step 2: Detect key -- top 7 most common pitch classes (diatonic scale)
        key_pitch_classes = set(
            pc for pc, _ in melody_pitch_classes.most_common(7)
        )

        # Step 3: Correct accompaniment notes
        corrected_accomp = []
        any_changed = False

        for beat in accomp_beats:
            if len(beat['tokens']) == 0:
                corrected_accomp.append(beat)
                continue

            notes = decode_beat(beat['tokens'])
            if len(notes) == 0:
                corrected_accomp.append(beat)
                continue

            corrected_notes = []
            beat_changed = False

            for abs_pitch, val in notes:
                if abs_pitch < 0 or abs_pitch > 87:
                    corrected_notes.append((abs_pitch, val))
                    continue

                midi_note = 108 - abs_pitch
                pc = midi_note % 12

                if pc not in key_pitch_classes:
                    # Find nearest in-key pitch by trying +1 and -1 semitone shifts
                    # In pitch_index space: +1 semitone = pitch_index - 1 (higher MIDI)
                    #                       -1 semitone = pitch_index + 1 (lower MIDI)
                    best_pitch = abs_pitch
                    for delta in [1, -1, 2, -2]:
                        candidate = abs_pitch - delta  # delta in MIDI semitones
                        if 0 <= candidate <= 87:
                            candidate_midi = 108 - candidate
                            candidate_pc = candidate_midi % 12
                            if candidate_pc in key_pitch_classes:
                                best_pitch = candidate
                                break

                    if best_pitch != abs_pitch:
                        corrected_notes.append((best_pitch, val))
                        beat_changed = True
                    else:
                        corrected_notes.append((abs_pitch, val))
                else:
                    corrected_notes.append((abs_pitch, val))

            if beat_changed:
                any_changed = True
                try:
                    new_tokens = encode_beat(corrected_notes)
                    new_beat = dict(beat)
                    new_beat['tokens'] = new_tokens
                    corrected_accomp.append(new_beat)
                except (AssertionError, ValueError):
                    # If encoding fails (e.g., pitch order violation), keep original
                    corrected_accomp.append(beat)
            else:
                corrected_accomp.append(beat)

        if not any_changed:
            return token_sequence

        # Step 4: Rebuild the token sequence
        return rebuild_interleaved(melody_beats, corrected_accomp, parsed)


def run_pipeline_on_file(
    npz_path: str,
    tagger_path: str,
    inserter_path: str,
    output_midi_path: Optional[str] = None,
    num_iterations: int = 2,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = 'cuda',
    mode: str = 'reedit',
):
    """
    Convenience function: NPZ file → FELIX pipeline → MIDI output.

    Args:
        npz_path: path to input npz file
        tagger_path: path to tagger checkpoint
        inserter_path: path to inserter checkpoint
        output_midi_path: path for output MIDI (auto-generated if None)
        mode: 'reedit' (perturb then fix) or 'generate' (blank accomp then generate)

    Returns:
        dict with output_tokens, output_midi_path, metrics (if reedit mode)
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    from data.dataset import FELIXBaseDataset
    from utils.token2midi import Token2MIDI
    from inference.evaluate import evaluate_reconstruction

    # Tokenize the npz file
    ds = FELIXBaseDataset([npz_path], data_dir='', max_len=4096)
    target_tokens = ds._tokenize_npz(0)

    # Create pipeline
    pipeline = FELIXPipeline(tagger_path, inserter_path, device=device)

    if mode == 'generate':
        output_tokens = pipeline.generate_from_melody(
            target_tokens, num_iterations, temperature, top_k
        )
    else:
        # Re-edit mode: perturb then try to reconstruct
        from data.perturbation import perturb_accompaniment
        from data.sequence_parser import parse_sequence, separate_tracks, rebuild_interleaved
        import copy

        parsed = parse_sequence(target_tokens)
        melody_beats, accomp_beats = separate_tracks(parsed)
        original_accomp = copy.deepcopy(accomp_beats)

        perturbed_accomp, level, _ = perturb_accompaniment(accomp_beats)
        source_tokens = rebuild_interleaved(melody_beats, perturbed_accomp, parsed)

        output_tokens = pipeline.generate(
            source_tokens, num_iterations, temperature, top_k
        )

    # Convert to MIDI
    if output_midi_path is None:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        output_midi_path = f'output_{base}_{mode}.mid'

    converter = Token2MIDI()
    converter.convert(output_tokens, output_midi_path)

    result = {
        'output_tokens': output_tokens,
        'output_midi_path': output_midi_path,
    }

    # Compute metrics for re-edit mode
    if mode == 'reedit':
        metrics = evaluate_reconstruction(output_tokens, target_tokens)
        result['metrics'] = metrics

    return result
