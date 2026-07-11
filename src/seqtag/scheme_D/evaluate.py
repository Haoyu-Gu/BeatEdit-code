"""
Music GECToR Evaluation Script (absolute_bundled encoding - Scheme D)

Computes metrics:
1. Edit F1 (precision, recall, F1 of non-KEEP labels)
2. Token Accuracy (exact match rate after applying edits)
3. Preservation Rate (correct KEEP predictions)
4. Beat Accuracy (fully correct beats)
5. Note Accuracy (bundled token correction rate)

Usage:
    python evaluate.py --checkpoint checkpoints/gector_absolute_bundled/best_model \
        --data_dir /path/to/data/npz --num_samples 1000
"""

import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from collections import defaultdict

from config import (
    NUM_LABELS, LABEL_KEEP, LABEL_DELETE, PAD_TOKEN, DATA_DIR,
    is_bundled_token, is_control_token,
    decode_label, TRAINING_DEFAULTS as TD,
)
from sequence_parser import parse_sequence, decode_beat, encode_beat
from perturbation import perturb_sequence
from label_extractor import extract_labels
from inference import load_model_for_inference, inference_single, post_process, apply_labels
from dataset import GECToRDataset, get_file_lists


def compute_metrics(source_tokens, target_tokens, predicted_tokens,
                    true_labels, predicted_labels):
    """
    Compute all evaluation metrics for a single sample.

    Args:
        source_tokens: perturbed input tokens
        target_tokens: ground truth clean tokens
        predicted_tokens: model output after inference
        true_labels: ground truth edit labels
        predicted_labels: model predicted labels (single round)

    Returns:
        dict of metrics
    """
    metrics = {}

    # --- Edit F1 ---
    tp = fp = fn = 0
    for tl, pl in zip(true_labels, predicted_labels):
        true_edit = (tl != LABEL_KEEP)
        pred_edit = (pl != LABEL_KEEP)
        if true_edit and pred_edit and tl == pl:
            tp += 1
        elif pred_edit and not true_edit:
            fp += 1
        elif true_edit and not pred_edit:
            fn += 1
        elif true_edit and pred_edit and tl != pl:
            fp += 1
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics['edit_precision'] = precision
    metrics['edit_recall'] = recall
    metrics['edit_f1'] = f1

    # --- Token Accuracy (after full inference) ---
    min_len = min(len(target_tokens), len(predicted_tokens))
    matches = sum(1 for a, b in zip(target_tokens[:min_len], predicted_tokens[:min_len])
                  if a == b)
    metrics['token_accuracy'] = matches / max(len(target_tokens), 1)
    metrics['sequence_exact_match'] = 1.0 if predicted_tokens == target_tokens else 0.0

    # --- Preservation Rate ---
    keep_correct = keep_total = 0
    for tl, pl in zip(true_labels, predicted_labels):
        if tl == LABEL_KEEP:
            keep_total += 1
            if pl == LABEL_KEEP:
                keep_correct += 1
    metrics['preservation_rate'] = keep_correct / max(keep_total, 1)

    # --- Beat Accuracy ---
    try:
        pred_info = parse_sequence(predicted_tokens)
        target_info = parse_sequence(target_tokens)

        beat_correct = beat_total = 0
        for pb, tb in zip(pred_info['beats'], target_info['beats']):
            beat_total += 1
            if pb['tokens'] == tb['tokens']:
                beat_correct += 1
        metrics['beat_accuracy'] = beat_correct / max(beat_total, 1)
    except Exception:
        metrics['beat_accuracy'] = 0.0

    # --- Note Accuracy (bundled token correction rate) ---
    note_correct = note_total = 0
    for i, (tl, pl) in enumerate(zip(true_labels, predicted_labels)):
        if i < len(source_tokens):
            tok = source_tokens[i]
            if is_bundled_token(tok):
                if tl != LABEL_KEEP:
                    note_total += 1
                    if pl == tl:
                        note_correct += 1

    metrics['note_accuracy'] = note_correct / max(note_total, 1)

    return metrics


def evaluate_model(model, data_dir, num_samples=1000, device='cpu',
                   max_iterations=2, keep_bias=0.3, error_threshold=0.5,
                   seed=42):
    """
    Full evaluation pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)

    _, _, test_files = get_file_lists(data_dir, seed=seed)
    test_files = test_files[:num_samples]

    ds = GECToRDataset(
        file_list=test_files,
        data_dir=data_dir,
        max_len=2048,
        include_clean=False,
    )

    all_metrics = defaultdict(list)
    num_evaluated = 0

    for idx in range(len(ds)):
        try:
            # Get original tokens
            original_tokens = ds._tokenize_npz(idx)
            if len(original_tokens) > 2048:
                original_tokens = original_tokens[:2048]

            # Generate perturbation
            source_tokens, target_tokens = perturb_sequence(original_tokens)

            # Extract ground truth labels
            true_labels = extract_labels(source_tokens, target_tokens)

            # Run inference
            predicted_tokens, info = inference_single(
                model, source_tokens, device=device,
                max_iterations=max_iterations,
                keep_confidence_bias=keep_bias,
                error_threshold=error_threshold,
            )

            # Get single-round predicted labels for metrics
            input_ids = torch.tensor([source_tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                detect_logits, tag_logits = model(input_ids)
            predicted_labels = tag_logits[0].argmax(dim=-1).cpu().tolist()

            # Compute metrics
            metrics = compute_metrics(
                source_tokens, target_tokens, predicted_tokens,
                true_labels, predicted_labels,
            )
            metrics['num_iterations'] = info['iterations']
            metrics['total_edits'] = sum(info['edits_per_round'])

            for k, v in metrics.items():
                all_metrics[k].append(v)
            num_evaluated += 1

            if (idx + 1) % 100 == 0:
                print(f"Evaluated {idx + 1}/{len(ds)} samples...")

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue

    # Aggregate
    results = {}
    for k, vals in all_metrics.items():
        results[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }

    results['num_evaluated'] = num_evaluated
    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("Music GECToR Evaluation Results (absolute_bundled)")
    print("=" * 60)

    key_metrics = [
        ('edit_f1', 'Edit F1'),
        ('edit_precision', 'Edit Precision'),
        ('edit_recall', 'Edit Recall'),
        ('token_accuracy', 'Token Accuracy'),
        ('sequence_exact_match', 'Sequence Exact Match'),
        ('preservation_rate', 'Preservation Rate'),
        ('beat_accuracy', 'Beat Accuracy'),
        ('note_accuracy', 'Note Accuracy'),
        ('num_iterations', 'Avg Iterations'),
    ]

    for key, name in key_metrics:
        if key in results:
            m = results[key]
            print(f"  {name:25s}: {m['mean']:.4f} +/- {m['std']:.4f}")

    print(f"\n  Samples evaluated: {results.get('num_evaluated', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Music GECToR Evaluation (absolute_bundled)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--max_iterations', type=int, default=TD['max_iterations'])
    parser.add_argument('--keep_bias', type=float, default=TD['keep_confidence_bias'])
    parser.add_argument('--error_threshold', type=float, default=TD['error_threshold'])
    parser.add_argument('--output', type=str, default=None,
                        help='Save results as JSON')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model = load_model_for_inference(args.checkpoint, args.device)
    print(f"Model loaded from {args.checkpoint}")

    results = evaluate_model(
        model,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        device=args.device,
        max_iterations=args.max_iterations,
        keep_bias=args.keep_bias,
        error_threshold=args.error_threshold,
        seed=args.seed,
    )

    print_results(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
