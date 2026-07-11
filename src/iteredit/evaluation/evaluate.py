"""
Evaluation script for Levenshtein Transformer Music Inpainting.

Computes objective and music quality metrics on test data:
  - Token-level accuracy (exact match in mask region)
  - Normalized edit distance (Levenshtein distance / target length)
  - Pitch accuracy (relative position component of bundled tokens)
  - Pattern accuracy (patch value component of bundled tokens)
  - Average refinement iterations
  - Note density ratio (inpainted vs context)
  - Length accuracy (inpainted length vs ground truth length)

Usage:
    cd /path/to/LevT_inpainting
    conda run -n musictoken python evaluation/evaluate.py \
        --checkpoint checkpoints/best.pt --n_samples 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import random
import time
import numpy as np
import torch

from configs.config import (
    BUNDLED_TOKEN_MIN, BUNDLED_TOKEN_MAX, PATTERN_NUM,
    EMPTY_MARKER, SPLIT_0, SPLIT_1, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    PLH_TOKEN, VOCAB_SIZE, DATA_DIR,
    is_bundled_token, is_control_token,
)
from data.levenshtein_utils import levenshtein_distance_and_path
from data.sequence_parser import parse_sequence, get_beat_boundaries
from data.dataset import LevTDataset, get_file_lists
from data.masking import create_inpainting_pair
from inference.pipeline import LevTInpaintingPipeline


# ==================== Metric Helpers ====================

def extract_mask_region(full_tokens, mask_start, mask_end):
    """
    Extract the ground-truth tokens in the mask region.

    Args:
        full_tokens: list of int, the complete original token sequence.
        mask_start: int, token index where the mask region starts (inclusive).
        mask_end: int, token index where the mask region ends (exclusive).

    Returns:
        list of int: the ground-truth tokens that were masked out.
    """
    return list(full_tokens[mask_start:mask_end])


def compute_metrics(predicted_tokens, ground_truth_tokens):
    """
    Compute all token-level metrics comparing predicted vs ground-truth
    for the inpainted (mask) region.

    Args:
        predicted_tokens: list of int, model output for the inpainted region.
        ground_truth_tokens: list of int, ground-truth tokens in the mask region.

    Returns:
        dict with keys:
            token_accuracy: float, fraction of tokens that match exactly.
            edit_distance: int, raw Levenshtein distance.
            normalized_edit_distance: float, edit_distance / max(len(ground_truth), 1).
            pitch_accuracy: float, fraction of bundled tokens where pitch matches.
            pattern_accuracy: float, fraction of bundled tokens where pattern matches.
            length_accuracy: float, len(predicted) / max(len(ground_truth), 1).
    """
    pred = list(predicted_tokens)
    gt = list(ground_truth_tokens)

    gt_len = len(gt)
    pred_len = len(pred)

    # --- Token-level accuracy (pad to same length for position-wise comparison) ---
    min_len = min(pred_len, gt_len)
    if min_len == 0:
        token_acc = 1.0 if pred_len == 0 and gt_len == 0 else 0.0
    else:
        matches = sum(1 for i in range(min_len) if pred[i] == gt[i])
        token_acc = matches / max(gt_len, 1)

    # --- Edit distance ---
    edit_dist, _ = levenshtein_distance_and_path(pred, gt)
    norm_edit_dist = edit_dist / max(gt_len, 1)

    # --- Pitch accuracy (relative position component: token // 81) ---
    # Only compare bundled tokens at matching positions
    pitch_matches = 0
    pitch_total = 0
    for i in range(min_len):
        if is_bundled_token(gt[i]):
            pitch_total += 1
            if is_bundled_token(pred[i]):
                gt_pitch = gt[i] // PATTERN_NUM
                pred_pitch = pred[i] // PATTERN_NUM
                if gt_pitch == pred_pitch:
                    pitch_matches += 1
    pitch_acc = pitch_matches / max(pitch_total, 1)

    # --- Pattern accuracy (patch value component: token % 81) ---
    pattern_matches = 0
    pattern_total = 0
    for i in range(min_len):
        if is_bundled_token(gt[i]):
            pattern_total += 1
            if is_bundled_token(pred[i]):
                gt_pattern = gt[i] % PATTERN_NUM
                pred_pattern = pred[i] % PATTERN_NUM
                if gt_pattern == pred_pattern:
                    pattern_matches += 1
    pattern_acc = pattern_matches / max(pattern_total, 1)

    # --- Length accuracy ---
    length_acc = pred_len / max(gt_len, 1)

    return {
        'token_accuracy': token_acc,
        'edit_distance': edit_dist,
        'normalized_edit_distance': norm_edit_dist,
        'pitch_accuracy': pitch_acc,
        'pattern_accuracy': pattern_acc,
        'length_accuracy': length_acc,
    }


def count_notes_in_region(tokens):
    """
    Count the number of bundled note tokens in a token list.

    Args:
        tokens: list of int.

    Returns:
        int: number of bundled note tokens (0-7127).
    """
    return sum(1 for t in tokens if is_bundled_token(t))


def compute_note_density_ratio(full_seq, mask_start, mask_end, predicted_region):
    """
    Compute the ratio of note density in the inpainted region vs the context region.

    A ratio close to 1.0 indicates the model produces music with similar density
    to the surrounding context.

    Args:
        full_seq: list of int, the original complete token sequence (ground truth).
        mask_start: int, start of mask region in full_seq.
        mask_end: int, end of mask region in full_seq (exclusive).
        predicted_region: list of int, the model's inpainted tokens.

    Returns:
        float: note_density_inpainted / note_density_context, or 0.0 if context is empty.
    """
    context_tokens = full_seq[:mask_start] + full_seq[mask_end:]
    context_notes = count_notes_in_region(context_tokens)
    context_len = len(context_tokens)

    pred_notes = count_notes_in_region(predicted_region)
    pred_len = len(predicted_region)

    if context_len == 0 or context_notes == 0:
        return 0.0

    context_density = context_notes / context_len
    pred_density = pred_notes / max(pred_len, 1)

    return pred_density / context_density


# ==================== Main Evaluation Loop ====================

def evaluate(args):
    """Run the full evaluation pipeline."""

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = 'cpu'

    print(f"Device: {args.device}")
    print(f"Loading model from: {args.checkpoint}")

    # Load the inference pipeline
    pipeline = LevTInpaintingPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded successfully.")

    # Get test file list
    print(f"Loading test files from: {args.data_dir}")
    _, _, test_files = get_file_lists(data_dir=args.data_dir, seed=args.seed)
    print(f"Total test files available: {len(test_files)}")

    n_samples = min(args.n_samples, len(test_files))
    test_files = test_files[:n_samples]
    print(f"Evaluating on {n_samples} samples (mask_ratio={args.mask_ratio})")

    # Create a dataset for tokenization only (no augmentation)
    tokenizer_ds = LevTDataset(
        file_list=test_files,
        data_dir=args.data_dir,
        max_len=2048,
        pitch_shift_augment=False,
    )

    # Collect per-sample results
    all_results = []
    failed = 0

    t_start = time.time()

    for sample_idx in range(n_samples):
        try:
            # 1. Tokenize NPZ to full token sequence
            full_tokens = tokenizer_ds._tokenize_npz(sample_idx)

            if len(full_tokens) > 2048:
                full_tokens = full_tokens[:2048]

            # 2. Parse sequence to find beats
            parsed = parse_sequence(full_tokens)
            beats = parsed['beats']
            num_beats = len(beats)

            if num_beats < 4:
                failed += 1
                continue

            # 3. Deterministic mask region based on mask_ratio
            mask_len = max(2, int(num_beats * args.mask_ratio))
            mask_len = min(mask_len, num_beats - 2)

            # Place mask in the middle for reproducible evaluation
            mask_beat_start = max(1, (num_beats - mask_len) // 2)
            mask_beat_end = mask_beat_start + mask_len

            # Convert beat indices to token indices
            tok_start = beats[mask_beat_start]['start_idx']
            if mask_beat_end < num_beats:
                tok_end = beats[mask_beat_end]['start_idx']
            else:
                if parsed['footer_tokens']:
                    tok_end = len(full_tokens) - len(parsed['footer_tokens'])
                else:
                    tok_end = len(full_tokens)

            # 4. Extract ground-truth mask region
            gt_region = extract_mask_region(full_tokens, tok_start, tok_end)
            if len(gt_region) == 0:
                failed += 1
                continue

            # 5. Create masked sequence (remove the mask region)
            masked_seq = full_tokens[:tok_start] + full_tokens[tok_end:]

            # 6. Run inference pipeline
            completed_seq, history = pipeline.inpaint(
                masked_seq,
                mask_start=tok_start,
                mask_end=tok_start,  # gap = single point after removal
                max_iter=args.max_iter,
                temperature=args.temperature,
                verbose=False,
            )

            # 7. Extract predicted mask region from completed sequence
            # The completed sequence has context_left + inpainted + context_right.
            # context_left is full_tokens[:tok_start], context_right is full_tokens[tok_end:]
            context_left_len = tok_start
            context_right_len = len(full_tokens) - tok_end
            predicted_region_end = len(completed_seq) - context_right_len
            predicted_region = completed_seq[context_left_len:predicted_region_end]

            # 8. Compute metrics
            metrics = compute_metrics(predicted_region, gt_region)

            # Number of iterations (history[0] is the initial state)
            n_iterations = len(history) - 1

            # Note density ratio
            density_ratio = compute_note_density_ratio(
                full_tokens, tok_start, tok_end, predicted_region
            )

            sample_result = {
                'sample_idx': sample_idx,
                'file': test_files[sample_idx],
                'full_seq_len': len(full_tokens),
                'num_beats': num_beats,
                'mask_beats': mask_len,
                'gt_region_len': len(gt_region),
                'pred_region_len': len(predicted_region),
                'n_iterations': n_iterations,
                'note_density_ratio': density_ratio,
                **metrics,
            }
            all_results.append(sample_result)

            if (sample_idx + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{sample_idx + 1}/{n_samples}] "
                      f"tok_acc={metrics['token_accuracy']:.4f} "
                      f"ned={metrics['normalized_edit_distance']:.4f} "
                      f"iters={n_iterations} "
                      f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"  Sample {sample_idx} failed: {e}")
            failed += 1
            continue

    elapsed_total = time.time() - t_start

    if len(all_results) == 0:
        print("No successful samples. Evaluation aborted.")
        return

    # ==================== Aggregate Results ====================

    agg = {}
    metric_keys = [
        'token_accuracy', 'normalized_edit_distance', 'edit_distance',
        'pitch_accuracy', 'pattern_accuracy', 'length_accuracy',
        'n_iterations', 'note_density_ratio',
    ]
    for key in metric_keys:
        values = [r[key] for r in all_results]
        agg[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }

    # ==================== Print Summary ====================

    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(f"Samples evaluated: {len(all_results)} / {n_samples} (failed: {failed})")
    print(f"Total time: {elapsed_total:.1f}s "
          f"({elapsed_total / max(len(all_results), 1):.2f}s per sample)")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Temperature: {args.temperature}")
    print("-" * 72)
    print(f"{'Metric':<32s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} "
          f"{'Min':>8s} {'Max':>8s}")
    print("-" * 72)

    display_names = {
        'token_accuracy':             'Token Accuracy',
        'normalized_edit_distance':   'Norm. Edit Distance',
        'edit_distance':              'Edit Distance (raw)',
        'pitch_accuracy':             'Pitch Accuracy',
        'pattern_accuracy':           'Pattern Accuracy',
        'length_accuracy':            'Length Accuracy',
        'n_iterations':               'Avg Iterations',
        'note_density_ratio':         'Note Density Ratio',
    }

    for key in metric_keys:
        name = display_names.get(key, key)
        m = agg[key]
        print(f"{name:<32s} {m['mean']:>8.4f} {m['std']:>8.4f} "
              f"{m['median']:>8.4f} {m['min']:>8.4f} {m['max']:>8.4f}")

    print("=" * 72)

    # ==================== Save Results ====================

    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        'args': vars(args),
        'summary': agg,
        'num_evaluated': len(all_results),
        'num_failed': failed,
        'elapsed_seconds': elapsed_total,
        'per_sample': all_results,
    }

    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Levenshtein Transformer Music Inpainting model."
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (required).',
    )
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR,
        help='NPZ data directory.',
    )
    parser.add_argument(
        '--n_samples', type=int, default=100,
        help='Number of test samples to evaluate (default: 100).',
    )
    parser.add_argument(
        '--max_iter', type=int, default=10,
        help='Max refinement iterations per sample (default: 10).',
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature for token prediction (default: 1.0).',
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.3,
        help='Fraction of beats to mask (default: 0.3).',
    )
    parser.add_argument(
        '--output_dir', type=str, default='evaluation/results/',
        help='Directory to save results JSON (default: evaluation/results/).',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device: cuda or cpu (default: cuda).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).',
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
