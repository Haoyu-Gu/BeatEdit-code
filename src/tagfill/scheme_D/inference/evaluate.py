"""
FELIX-Music Evaluation Module.

Metrics:
1. Reconstruction accuracy (perturb → reconstruct, compare to original)
2. Musical quality metrics (pitch class overlap, rhythm similarity, note density)
3. Tagger-specific metrics (per-label precision/recall/F1)
"""

import os
import random
import collections
import numpy as np
from typing import List, Dict, Optional

from configs.config import (
    NUM_FELIX_LABELS, LABEL_KEEP, LABEL_PAD, decode_felix_label,
)
from data.sequence_parser import (
    parse_sequence, separate_tracks, decode_beat,
)


def beat_exact_match(pred_accomp_beats, target_accomp_beats):
    """
    Compute beat-level exact match rate.

    Args:
        pred_accomp_beats: list of beat dicts (predicted)
        target_accomp_beats: list of beat dicts (ground truth)

    Returns:
        exact_match_rate: fraction of beats that are exactly identical
    """
    n = min(len(pred_accomp_beats), len(target_accomp_beats))
    if n == 0:
        return 0.0

    matches = 0
    for i in range(n):
        if pred_accomp_beats[i]['tokens'] == target_accomp_beats[i]['tokens']:
            matches += 1

    return matches / n


def token_level_accuracy(pred_accomp_beats, target_accomp_beats):
    """
    Compute token-level accuracy across all accompaniment beats.

    Compares decoded note lists (pitch, value) per beat.
    """
    total_notes = 0
    correct_notes = 0

    n = min(len(pred_accomp_beats), len(target_accomp_beats))
    for i in range(n):
        pred_notes = set(decode_beat(pred_accomp_beats[i]['tokens']))
        tgt_notes = set(decode_beat(target_accomp_beats[i]['tokens']))

        if len(tgt_notes) == 0 and len(pred_notes) == 0:
            total_notes += 1
            correct_notes += 1
        else:
            total_notes += max(len(tgt_notes), 1)
            correct_notes += len(pred_notes & tgt_notes)

    return correct_notes / max(total_notes, 1)


def pitch_class_histogram(accomp_beats):
    """
    Compute pitch class distribution (12 classes) from accompaniment beats.

    Returns:
        histogram: numpy array of shape (12,), normalized to sum to 1
    """
    hist = np.zeros(12, dtype=np.float64)

    for beat in accomp_beats:
        notes = decode_beat(beat['tokens'])
        for pitch, val in notes:
            # pitch is 0-87, map to MIDI pitch = pitch + 21
            pc = (pitch + 21) % 12
            hist[pc] += 1

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_class_overlap(pred_beats, target_beats):
    """
    Compute overlap between pitch class distributions.

    Returns a value in [0, 1] where 1 = identical distributions.
    Uses histogram intersection (sum of min).
    """
    pred_hist = pitch_class_histogram(pred_beats)
    tgt_hist = pitch_class_histogram(target_beats)
    return np.minimum(pred_hist, tgt_hist).sum()


def rhythm_pattern_distribution(accomp_beats):
    """
    Collect distribution of patch values (rhythm patterns) across beats.

    Returns:
        Counter mapping pattern_value → count
    """
    dist = collections.Counter()
    for beat in accomp_beats:
        notes = decode_beat(beat['tokens'])
        for pitch, val in notes:
            dist[val] += 1
    return dist


def rhythm_similarity(pred_beats, target_beats):
    """
    Compute cosine similarity between rhythm pattern distributions.
    """
    pred_dist = rhythm_pattern_distribution(pred_beats)
    tgt_dist = rhythm_pattern_distribution(target_beats)

    all_keys = set(pred_dist.keys()) | set(tgt_dist.keys())
    if not all_keys:
        return 1.0

    pred_vec = np.array([pred_dist.get(k, 0) for k in sorted(all_keys)], dtype=np.float64)
    tgt_vec = np.array([tgt_dist.get(k, 0) for k in sorted(all_keys)], dtype=np.float64)

    dot = np.dot(pred_vec, tgt_vec)
    norm = np.linalg.norm(pred_vec) * np.linalg.norm(tgt_vec)
    if norm == 0:
        return 0.0
    return dot / norm


def note_density(accomp_beats):
    """Compute average number of notes per beat."""
    if len(accomp_beats) == 0:
        return 0.0
    total = sum(len(decode_beat(b['tokens'])) for b in accomp_beats)
    return total / len(accomp_beats)


def note_density_ratio(pred_beats, target_beats):
    """
    Compute ratio of predicted to target note density.
    Ideal = 1.0. >1 means more notes, <1 means fewer.
    """
    pred_density = note_density(pred_beats)
    tgt_density = note_density(target_beats)
    if tgt_density == 0:
        return float('inf') if pred_density > 0 else 1.0
    return pred_density / tgt_density


def evaluate_reconstruction(pred_tokens, target_tokens):
    """
    Full reconstruction evaluation: compare predicted output to target.

    Args:
        pred_tokens: predicted token sequence (list of ints)
        target_tokens: ground truth token sequence (list of ints)

    Returns:
        dict of metrics
    """
    pred_parsed = parse_sequence(pred_tokens)
    tgt_parsed = parse_sequence(target_tokens)

    _, pred_accomp = separate_tracks(pred_parsed)
    _, tgt_accomp = separate_tracks(tgt_parsed)

    return {
        'beat_exact_match': beat_exact_match(pred_accomp, tgt_accomp),
        'token_accuracy': token_level_accuracy(pred_accomp, tgt_accomp),
        'pitch_class_overlap': pitch_class_overlap(pred_accomp, tgt_accomp),
        'rhythm_similarity': rhythm_similarity(pred_accomp, tgt_accomp),
        'note_density_ratio': note_density_ratio(pred_accomp, tgt_accomp),
        'pred_note_density': note_density(pred_accomp),
        'target_note_density': note_density(tgt_accomp),
    }


def evaluate_tagger_predictions(pred_labels, true_labels):
    """
    Evaluate Tagger predictions.

    Args:
        pred_labels: list of predicted label IDs
        true_labels: list of ground truth label IDs

    Returns:
        dict with overall and per-label metrics
    """
    pred = np.array(pred_labels)
    true = np.array(true_labels)

    # Filter out padding
    valid = true != LABEL_PAD
    pred = pred[valid]
    true = true[valid]

    if len(true) == 0:
        return {'accuracy': 0.0, 'per_label': {}}

    accuracy = (pred == true).mean()

    per_label = {}
    for lid in range(NUM_FELIX_LABELS):
        mask = true == lid
        if mask.sum() == 0:
            continue
        op, val = decode_felix_label(lid)
        tp = ((pred == lid) & (true == lid)).sum()
        fp = ((pred == lid) & (true != lid)).sum()
        fn = ((pred != lid) & (true == lid)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_label[f'{op}({val})'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(mask.sum()),
        }

    return {
        'accuracy': float(accuracy),
        'per_label': per_label,
    }


def print_evaluation_report(metrics, title="Evaluation Report"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    for key, value in metrics.items():
        if key == 'per_label':
            print(f"\n  Per-label metrics:")
            for label, m in sorted(value.items(), key=lambda x: -x[1]['support']):
                print(f"    {label:>25s}: P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} (n={m['support']})")
        elif isinstance(value, float):
            print(f"  {key:>25s}: {value:.4f}")
        else:
            print(f"  {key:>25s}: {value}")

    print(f"{'='*60}\n")
