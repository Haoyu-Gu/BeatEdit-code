"""
REMI GECToR End-to-End Evaluation

Evaluates REMI GECToR on test set:
1. Load MIDI -> tokenize to REMI
2. Perturb (create source)
3. Run inference (correct source)
4. Compare corrected vs original at note level

Metrics: note_f1_tol{0,2,4}, chroma_f1, mean_pitch_error, beat_exact_match
Output: results/baselines/remi_gector_eval.json

Usage:
    python evaluate.py --checkpoint checkpoints/gector_remi_stage3/best_model \
        --num_samples 200 --gpu 2
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
from collections import defaultdict

from config import (
    MIDI_DATA_DIR, BAR_TOKEN, BOS_TOKEN, EOS_TOKEN,
    LABEL_KEEP, POSITION_OFFSET, PITCH_OFFSET, MIDI_PITCH_MIN,
)
from remi_tokenizer import midi_to_tokens, tokens_to_notes
from perturbation import perturb_sequence, perturb_sequence_per_position
from inference import load_model_for_inference, inference_single
from dataset import get_file_lists


def note_level_comparison(pred_notes, target_notes, tolerance=0):
    """
    Compare predicted and target note lists.

    Each note is (bar, position, midi_pitch, velocity_token, duration_token).
    Two notes match if bar, position, and pitch are within tolerance.

    Returns: (tp, fp, fn, pitch_errors)
    """
    # Convert to comparable tuples
    pred_set = [(n['bar'], n['position'], n['pitch']) for n in pred_notes]
    target_set = [(n['bar'], n['position'], n['pitch']) for n in target_notes]

    # Match with tolerance
    matched_pred = set()
    matched_target = set()
    pitch_errors = []

    for ti, (tb, tp_, tpitch) in enumerate(target_set):
        best_dist = tolerance + 1
        best_pi = None
        for pi, (pb, pp, ppitch) in enumerate(pred_set):
            if pi in matched_pred:
                continue
            if pb == tb and pp == tp_:
                dist = abs(ppitch - tpitch)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_pi = pi
        if best_pi is not None:
            matched_pred.add(best_pi)
            matched_target.add(ti)
            pitch_errors.append(best_dist)

    tp = len(matched_target)
    fp = len(pred_set) - len(matched_pred)
    fn = len(target_set) - len(matched_target)

    return tp, fp, fn, pitch_errors


def chroma_comparison(pred_notes, target_notes):
    """
    Compare chroma (pitch class) distributions.
    Returns F1 score based on pitch class matching.
    """
    pred_chromas = [(n['bar'], n['position'], n['pitch'] % 12) for n in pred_notes]
    target_chromas = [(n['bar'], n['position'], n['pitch'] % 12) for n in target_notes]

    pred_set = set(pred_chromas)
    target_set = set(target_chromas)

    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1


def beat_exact_match(pred_tokens, target_tokens):
    """
    Compute bar-level exact match between predicted and target tokens.

    Extracts content per bar and checks if bars are identical.
    """
    def get_bar_contents(tokens):
        bars = []
        current_bar = []
        in_bar = False
        for t in tokens:
            if t == BAR_TOKEN:
                if in_bar:
                    bars.append(tuple(current_bar))
                current_bar = []
                in_bar = True
            elif in_bar and t not in (BOS_TOKEN, EOS_TOKEN):
                current_bar.append(t)
        if in_bar:
            bars.append(tuple(current_bar))
        return bars

    pred_bars = get_bar_contents(pred_tokens)
    target_bars = get_bar_contents(target_tokens)

    if len(pred_bars) == 0 and len(target_bars) == 0:
        return 1.0

    matches = 0
    total = max(len(pred_bars), len(target_bars))
    for i in range(min(len(pred_bars), len(target_bars))):
        if pred_bars[i] == target_bars[i]:
            matches += 1

    return matches / max(total, 1)


def evaluate_single(model, midi_path, device, p_pitch=0.10, p_rhythm=0.05,
                    p_delete=0.03, p_insert=0.02, seed=None,
                    perturb_mode='bar'):
    """Evaluate a single MIDI file.

    Args:
        perturb_mode: 'bar' (default, per-bar perturbation) or
                      'position' (per-position, matching BEAT's per-beat intensity)
    """
    if seed is not None:
        random.seed(seed)

    # 1. Tokenize original
    target_tokens = midi_to_tokens(midi_path)
    if len(target_tokens) == 0:
        return None

    if target_tokens[0] != BOS_TOKEN:
        target_tokens = [BOS_TOKEN] + target_tokens
    if target_tokens[-1] != EOS_TOKEN:
        target_tokens.append(EOS_TOKEN)

    # Truncate to 2048
    if len(target_tokens) > 2048:
        target_tokens = target_tokens[:2048]

    # 2. Perturb
    perturb_fn = perturb_sequence_per_position if perturb_mode == 'position' else perturb_sequence
    source_tokens, _ = perturb_fn(
        target_tokens,
        p_pitch=p_pitch,
        p_rhythm=p_rhythm,
        p_delete=p_delete,
        p_insert=p_insert,
    )

    # 3. Inference
    corrected_tokens, info = inference_single(model, source_tokens, device=device)

    # 4. Extract notes
    target_notes = tokens_to_notes(target_tokens)
    pred_notes = tokens_to_notes(corrected_tokens)
    source_notes = tokens_to_notes(source_tokens)

    # 5. Metrics
    results = {}

    # Note F1 at different tolerances
    for tol in [0, 2, 4]:
        tp, fp, fn, pitch_errors = note_level_comparison(pred_notes, target_notes, tolerance=tol)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        results[f'note_f1_tol{tol}'] = f1

    # Chroma F1
    results['chroma_f1'] = chroma_comparison(pred_notes, target_notes)

    # Mean pitch error (for matched notes)
    _, _, _, pitch_errors = note_level_comparison(pred_notes, target_notes, tolerance=12)
    results['mean_pitch_error'] = np.mean(pitch_errors) if pitch_errors else 0.0

    # Beat/bar exact match
    results['beat_exact_match'] = beat_exact_match(corrected_tokens, target_tokens)

    # No-Edit baseline (source vs target)
    results['noedit_beat_match'] = beat_exact_match(source_tokens, target_tokens)

    # Metadata
    results['num_target_notes'] = len(target_notes)
    results['num_pred_notes'] = len(pred_notes)
    results['num_source_notes'] = len(source_notes)
    results['target_len'] = len(target_tokens)
    results['source_len'] = len(source_tokens)
    results['corrected_len'] = len(corrected_tokens)
    results['iterations'] = info['iterations']
    results['total_edits'] = sum(info['edits_per_round'])

    return results


def main():
    parser = argparse.ArgumentParser(description='REMI GECToR Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default=MIDI_DATA_DIR)
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'results', 'baselines',
                                             'remi_gector_eval.json'))
    parser.add_argument('--perturb_mode', type=str, default='bar',
                        choices=['bar', 'position'],
                        help='bar=per-bar (easy), position=per-position (BEAT-compatible)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model_for_inference(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    # Get test files
    _, _, test_files = get_file_lists(args.data_dir, seed=args.seed)
    if args.num_samples < len(test_files):
        rng = random.Random(args.seed)
        test_files = rng.sample(test_files, args.num_samples)
    print(f"Evaluating on {len(test_files)} test files (perturb_mode={args.perturb_mode})")

    # Evaluate
    all_results = []
    for i, fname in enumerate(test_files):
        fpath = os.path.join(args.data_dir, fname)
        try:
            result = evaluate_single(model, fpath, device, seed=args.seed + i,
                                        perturb_mode=args.perturb_mode)
            if result is not None:
                result['file'] = fname
                all_results.append(result)
        except Exception as e:
            print(f"  Error on {fname}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_files)}")

    # Aggregate
    metrics = defaultdict(list)
    for r in all_results:
        for k in ['note_f1_tol0', 'note_f1_tol2', 'note_f1_tol4',
                   'chroma_f1', 'mean_pitch_error', 'beat_exact_match',
                   'noedit_beat_match']:
            metrics[k].append(r[k])

    print("\n" + "=" * 60)
    print(f"REMI GECToR Evaluation Results ({len(all_results)} samples)")
    print("=" * 60)
    for k, vals in sorted(metrics.items()):
        print(f"  {k:25s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")
    print("=" * 60)

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = {
        'num_samples': len(all_results),
        'metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                    for k, v in metrics.items()},
        'per_sample': all_results,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
