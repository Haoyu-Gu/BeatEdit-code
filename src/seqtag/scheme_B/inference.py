"""
Music GECToR Inference Pipeline

Multi-round iterative inference with:
- KEEP confidence bias
- Error detection thresholding
- Post-processing for encoding validity

Usage:
    python inference.py --checkpoint checkpoints/gector/best_model \
        --input test_corrupted.txt --output corrected.txt
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Optional

from config import (
    NUM_LABELS, LABEL_KEEP, PAD_TOKEN, VOCAB_SIZE,
    EMPTY_MARKER, END_MARKER, BAR_TOKEN,
    is_control_token, is_position_token, is_patch_value, is_music_token,
    decode_label, TRAINING_DEFAULTS as TD,
)
from sequence_parser import parse_sequence, decode_beat, encode_beat, reassemble_sequence
from model import MusicGECToR


def apply_labels(tokens, labels):
    """
    Apply predicted labels to token sequence.

    APPEND inserts after current token.
    DELETE removes the token.
    REPLACE/SHIFT modifies the token.
    """
    result = []

    for token, label in zip(tokens, labels):
        op, value = decode_label(label)

        if op == 'KEEP':
            result.append(token)
        elif op == 'DELETE':
            pass  # skip
        elif op == 'REPLACE':
            result.append(value)
        elif op == 'SHIFT':
            new_token = token + value
            if is_position_token(new_token):
                result.append(new_token)
            else:
                result.append(token)  # out of range → keep original
        elif op == 'APPEND':
            result.append(token)    # keep current
            result.append(value)    # add new token after

    return result


def post_process(tokens):
    """
    Post-process edited tokens to ensure encoding validity.

    For each beat:
    1. Decode to (abs_pitch, val) notes
    2. Filter invalid notes
    3. Re-encode with correct relative positions
    """
    try:
        beats_info = parse_sequence(tokens)
    except Exception:
        return tokens  # can't parse, return as-is

    processed_beats = []
    for beat in beats_info['beats']:
        try:
            notes = decode_beat(beat['tokens'])
            # Filter invalid notes
            notes = [(p, v) for p, v in notes
                     if 0 <= p <= 87 and 0 <= v <= 80]
            # Remove duplicate pitches (keep first)
            seen = set()
            unique_notes = []
            for p, v in notes:
                if p not in seen:
                    seen.add(p)
                    unique_notes.append((p, v))
            processed_beats.append(encode_beat(unique_notes))
        except (AssertionError, IndexError, Exception):
            # Decode failed → keep original tokens
            processed_beats.append(beat['tokens'])

    return reassemble_sequence(beats_info, processed_beats)


def inference_single(model, input_tokens, device='cpu',
                     max_iterations=3, keep_confidence_bias=0.3,
                     error_threshold=0.5):
    """
    GECToR-style iterative inference for a single sequence.

    Args:
        model: MusicGECToR model
        input_tokens: input token sequence (list of ints)
        device: torch device
        max_iterations: maximum correction rounds
        keep_confidence_bias: positive bias added to KEEP logit
        error_threshold: minimum error detection probability to apply correction

    Returns:
        corrected_tokens: corrected token sequence
        info: dict with per-iteration statistics
    """
    model.eval()
    current_tokens = list(input_tokens)
    info = {'iterations': 0, 'edits_per_round': []}

    for iteration in range(max_iterations):
        input_ids = torch.tensor([current_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            detect_logits, tag_logits = model(input_ids, attention_mask)

        detect_probs = torch.softmax(detect_logits[0], dim=-1)  # (seq_len, 2)
        tag_logits_seq = tag_logits[0]  # (seq_len, num_labels)

        # Apply KEEP confidence bias
        tag_logits_seq[:, LABEL_KEEP] += keep_confidence_bias

        # Determine labels
        predicted_labels = []
        num_edits = 0

        for i in range(len(current_tokens)):
            # Control tokens always KEEP
            if is_control_token(current_tokens[i]):
                predicted_labels.append(LABEL_KEEP)
                continue

            # Error detection gate
            error_prob = detect_probs[i, 1].item()
            if error_prob < error_threshold:
                predicted_labels.append(LABEL_KEEP)
            else:
                label = tag_logits_seq[i].argmax().item()
                predicted_labels.append(label)
                if label != LABEL_KEEP:
                    num_edits += 1

        info['edits_per_round'].append(num_edits)
        info['iterations'] = iteration + 1

        # No edits → converged
        if num_edits == 0:
            break

        # Apply labels
        new_tokens = apply_labels(current_tokens, predicted_labels)

        # Post-process
        new_tokens = post_process(new_tokens)

        # Check for oscillation
        if new_tokens == current_tokens:
            break

        current_tokens = new_tokens

    return current_tokens, info


def inference_batch(model, token_sequences, device='cpu', batch_size=32,
                    max_iterations=3, keep_confidence_bias=0.3,
                    error_threshold=0.5):
    """
    Batch inference for multiple sequences.

    Note: Since sequences change length between iterations,
    we process each iteration as a full batch, then iterate.
    """
    results = []
    infos = []

    for i in range(0, len(token_sequences), batch_size):
        batch_seqs = token_sequences[i:i + batch_size]
        batch_results = []
        batch_infos = []

        for seq in batch_seqs:
            corrected, info = inference_single(
                model, seq, device=device,
                max_iterations=max_iterations,
                keep_confidence_bias=keep_confidence_bias,
                error_threshold=error_threshold,
            )
            batch_results.append(corrected)
            batch_infos.append(info)

        results.extend(batch_results)
        infos.extend(batch_infos)

    return results, infos


def load_model_for_inference(checkpoint_path, device='cpu'):
    """Load trained MusicGECToR model for inference."""
    model = MusicGECToR(num_labels=NUM_LABELS)

    model_path = os.path.join(checkpoint_path, 'model.pt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Music GECToR Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (one token sequence per line, space-separated)')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_iterations', type=int, default=TD['max_iterations'])
    parser.add_argument('--keep_bias', type=float, default=TD['keep_confidence_bias'])
    parser.add_argument('--error_threshold', type=float, default=TD['error_threshold'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = load_model_for_inference(args.checkpoint, args.device)
    print(f"Model loaded from {args.checkpoint}")

    # Read input
    sequences = []
    with open(args.input, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            sequences.append(tokens)
    print(f"Loaded {len(sequences)} sequences")

    # Run inference
    results, infos = inference_batch(
        model, sequences, device=args.device,
        max_iterations=args.max_iterations,
        keep_confidence_bias=args.keep_bias,
        error_threshold=args.error_threshold,
    )

    # Write output
    with open(args.output, 'w') as f:
        for tokens in results:
            f.write(' '.join(map(str, tokens)) + '\n')

    # Statistics
    avg_iters = np.mean([info['iterations'] for info in infos])
    avg_edits = np.mean([sum(info['edits_per_round']) for info in infos])
    print(f"Results written to {args.output}")
    print(f"Avg iterations: {avg_iters:.2f}, Avg total edits: {avg_edits:.1f}")


if __name__ == '__main__':
    main()
