"""
Unified Metrics Module.

All evaluation metrics in one place. Every other script imports from here.

Metrics (paper main table):
  1. beat_exact_match  — per-beat token sequence exact match
  2. note_f1           — note-level F1 with pitch tolerance
  3. mean_pitch_error  — average absolute pitch distance (semitones)
  4. fmd               — Frechet Music Distance (BERT embedding)

Diagnostics (appendix):
  5. chroma_f1         — pitch class F1 ignoring octave
  6. rhythm_similarity — patch value distribution cosine similarity
  7. token_accuracy    — decoded note set intersection
  8. context_preservation — non-edited region preservation rate
"""

import numpy as np
import collections
from scipy import linalg as scipy_linalg
from sklearn.covariance import LedoitWolf


# ==================== Beat-level exact match ====================

def beat_exact_match(pred_beats, tgt_beats):
    """
    Beat-level token sequence exact match rate.

    Args:
        pred_beats: list of beat dicts with 'tokens' key
        tgt_beats: list of beat dicts with 'tokens' key

    Returns:
        float: fraction of beats that are exactly identical
    """
    n = min(len(pred_beats), len(tgt_beats))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if pred_beats[i]['tokens'] == tgt_beats[i]['tokens'])
    return matches / n


# ==================== Note-level F1 ====================

def note_f1(pred_beats, tgt_beats, decode_fn, tol=0):
    """
    Note-level F1 with pitch tolerance.

    For each beat, extract notes as (pitch, value) pairs.
    A predicted note matches a target note if |pred_pitch - tgt_pitch| <= tol.
    Uses greedy closest-first matching to avoid double-counting.

    Args:
        pred_beats: list of beat dicts
        tgt_beats: list of beat dicts
        decode_fn: callable(tokens) -> list of (abs_pitch, patch_value)
        tol: pitch tolerance in semitones (0 = exact)

    Returns:
        float: F1 score
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    n = min(len(pred_beats), len(tgt_beats))
    for i in range(n):
        pred_notes = decode_fn(pred_beats[i]['tokens'])
        tgt_notes = decode_fn(tgt_beats[i]['tokens'])

        pred_pitches = [p for p, v in pred_notes]
        tgt_pitches = [p for p, v in tgt_notes]

        # Greedy matching: match closest pairs first
        matched_pred = set()
        matched_tgt = set()

        pairs = []
        for ti, tp in enumerate(tgt_pitches):
            for pi, pp in enumerate(pred_pitches):
                d = abs(tp - pp)
                if d <= tol:
                    pairs.append((d, ti, pi))
        pairs.sort()

        for d, ti, pi in pairs:
            if ti not in matched_tgt and pi not in matched_pred:
                matched_tgt.add(ti)
                matched_pred.add(pi)
                total_tp += 1

        total_fn += len(tgt_pitches) - len(matched_tgt)
        total_fp += len(pred_pitches) - len(matched_pred)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ==================== Mean Pitch Error ====================

def mean_pitch_error(pred_beats, tgt_beats, decode_fn):
    """
    Average absolute pitch distance between best-matched note pairs.
    Lower is better (0 = perfect).

    Args:
        pred_beats, tgt_beats: lists of beat dicts
        decode_fn: callable(tokens) -> list of (abs_pitch, patch_value)

    Returns:
        float: mean absolute pitch error in semitones
    """
    total_error = 0.0
    total_count = 0

    n = min(len(pred_beats), len(tgt_beats))
    for i in range(n):
        pred_notes = decode_fn(pred_beats[i]['tokens'])
        tgt_notes = decode_fn(tgt_beats[i]['tokens'])

        pred_pitches = sorted([p for p, v in pred_notes])
        tgt_pitches = sorted([p for p, v in tgt_notes])

        # Match by sorted order (pitch-aligned)
        k = min(len(pred_pitches), len(tgt_pitches))
        for j in range(k):
            total_error += abs(pred_pitches[j] - tgt_pitches[j])
            total_count += 1

        # Unmatched notes count as max error (88 keys range)
        unmatched = abs(len(pred_pitches) - len(tgt_pitches))
        total_error += unmatched * 44  # half-range as penalty
        total_count += unmatched

    return total_error / max(total_count, 1)


# ==================== Chroma F1 ====================

def chroma_f1(pred_beats, tgt_beats, decode_fn):
    """
    F1 on pitch class (chroma) sets per beat.
    Ignores octave — only checks if the right note name (C, C#, ...) is present.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    n = min(len(pred_beats), len(tgt_beats))
    for i in range(n):
        pred_notes = decode_fn(pred_beats[i]['tokens'])
        tgt_notes = decode_fn(tgt_beats[i]['tokens'])

        # pitch 0-87, MIDI = pitch+21, chroma = (pitch+21)%12
        pred_chromas = set((p + 21) % 12 for p, v in pred_notes)
        tgt_chromas = set((p + 21) % 12 for p, v in tgt_notes)

        tp = len(pred_chromas & tgt_chromas)
        fp = len(pred_chromas - tgt_chromas)
        fn = len(tgt_chromas - pred_chromas)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ==================== Rhythm Similarity ====================

def rhythm_similarity(pred_beats, tgt_beats, decode_fn):
    """Cosine similarity between rhythm pattern (patch value) distributions."""
    pred_dist = collections.Counter()
    tgt_dist = collections.Counter()

    n = min(len(pred_beats), len(tgt_beats))
    for i in range(n):
        for p, v in decode_fn(pred_beats[i]['tokens']):
            pred_dist[v] += 1
        for p, v in decode_fn(tgt_beats[i]['tokens']):
            tgt_dist[v] += 1

    all_keys = set(pred_dist.keys()) | set(tgt_dist.keys())
    if not all_keys:
        return 1.0

    pred_vec = np.array([pred_dist.get(k, 0) for k in sorted(all_keys)], dtype=np.float64)
    tgt_vec = np.array([tgt_dist.get(k, 0) for k in sorted(all_keys)], dtype=np.float64)

    dot = np.dot(pred_vec, tgt_vec)
    norm = np.linalg.norm(pred_vec) * np.linalg.norm(tgt_vec)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ==================== Token Accuracy ====================

def token_accuracy(pred_beats, tgt_beats, decode_fn):
    """Decoded note set intersection metric."""
    total = 0
    correct = 0
    n = min(len(pred_beats), len(tgt_beats))
    for i in range(n):
        pred_notes = set(decode_fn(pred_beats[i]['tokens']))
        tgt_notes = set(decode_fn(tgt_beats[i]['tokens']))
        if len(tgt_notes) == 0 and len(pred_notes) == 0:
            total += 1
            correct += 1
        else:
            total += max(len(tgt_notes), 1)
            correct += len(pred_notes & tgt_notes)
    return correct / max(total, 1)


# ==================== Context Preservation ====================

def context_preservation(pred_beats, tgt_beats, changed_mask):
    """
    Preservation rate of non-edited regions.
    Only checks beats where changed_mask[i] is False.

    Args:
        pred_beats, tgt_beats: lists of beat dicts
        changed_mask: list of booleans (True = this beat was perturbed)

    Returns:
        float: fraction of unchanged beats that remain identical
    """
    total = 0
    preserved = 0
    n = min(len(pred_beats), len(tgt_beats), len(changed_mask))
    for i in range(n):
        if not changed_mask[i]:
            total += 1
            if pred_beats[i]['tokens'] == tgt_beats[i]['tokens']:
                preserved += 1
    return preserved / max(total, 1)


# ==================== Frechet Music Distance ====================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet Distance between two multivariate Gaussians.
    FMD = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrtm(sigma1 @ sigma2))
    """
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    diff = mu1 - mu2
    covmean, _ = scipy_linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy_linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_fmd(pred_embeddings, ref_embeddings):
    """
    Compute FMD from lists of embedding vectors.

    Args:
        pred_embeddings: list/array of shape (N, D) — predicted embeddings
        ref_embeddings: list/array of shape (N, D) — reference embeddings

    Returns:
        float: FMD value (lower = better)
    """
    pred_emb = np.array(pred_embeddings)
    ref_emb = np.array(ref_embeddings)

    if len(pred_emb) < 2 or len(ref_emb) < 2:
        return float('nan')

    mu_pred = np.mean(pred_emb, axis=0)
    mu_ref = np.mean(ref_emb, axis=0)
    sigma_pred = LedoitWolf().fit(pred_emb).covariance_
    sigma_ref = LedoitWolf().fit(ref_emb).covariance_

    return calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)


# ==================== Aggregate helpers ====================

# Keys used in paper main table
MAIN_METRIC_KEYS = [
    'beat_exact_match', 'note_f1_tol0', 'note_f1_tol2', 'note_f1_tol4',
    'chroma_f1', 'mean_pitch_error',
]

# All keys including diagnostics
ALL_METRIC_KEYS = MAIN_METRIC_KEYS + [
    'token_accuracy', 'rhythm_similarity', 'context_preservation',
    'bert_cosine_sim',
]


def compute_all_metrics(pred_beats, tgt_beats, decode_fn, changed_mask=None):
    """
    Compute all metrics on accompaniment beat lists.

    Args:
        pred_beats: predicted accompaniment beats
        tgt_beats: target accompaniment beats
        decode_fn: callable(tokens) -> list of (abs_pitch, patch_value)
        changed_mask: optional list of bools for context_preservation

    Returns:
        dict of metric_name -> value
    """
    result = {
        'beat_exact_match': beat_exact_match(pred_beats, tgt_beats),
        'token_accuracy': token_accuracy(pred_beats, tgt_beats, decode_fn),
        'note_f1_tol0': note_f1(pred_beats, tgt_beats, decode_fn, tol=0),
        'note_f1_tol2': note_f1(pred_beats, tgt_beats, decode_fn, tol=2),
        'note_f1_tol4': note_f1(pred_beats, tgt_beats, decode_fn, tol=4),
        'chroma_f1': chroma_f1(pred_beats, tgt_beats, decode_fn),
        'mean_pitch_error': mean_pitch_error(pred_beats, tgt_beats, decode_fn),
        'rhythm_similarity': rhythm_similarity(pred_beats, tgt_beats, decode_fn),
    }

    if changed_mask is not None:
        result['context_preservation'] = context_preservation(
            pred_beats, tgt_beats, changed_mask
        )

    return result


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: array-like of sample values
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        (ci_lower, ci_upper): tuple of floats
    """
    values = np.array(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    if n == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (ci_lower, ci_upper)


def safe_aggregate(metrics_list, keys=None, include_per_sample=False,
                   include_ci=True, n_bootstrap=1000, ci_level=0.95):
    """
    Aggregate a list of per-sample metric dicts into summary statistics.

    Args:
        metrics_list: list of dicts, each with metric_name -> float
        keys: which metric keys to aggregate (default: ALL_METRIC_KEYS)
        include_per_sample: if True, include raw per-sample values in output
        include_ci: if True, compute bootstrap 95% CI and normal CI
        n_bootstrap: number of bootstrap resamples for CI
        ci_level: confidence level (default 0.95)

    Returns:
        dict of metric_name -> {mean, std, median, n, ci95_boot, ci95_normal, [per_sample]}
    """
    if keys is None:
        keys = ALL_METRIC_KEYS
    agg = {}
    for key in keys:
        values = [m.get(key) for m in metrics_list
                  if m.get(key) is not None and np.isfinite(m.get(key, float('nan')))]
        if values:
            n = len(values)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            entry = {
                'mean': round(mean_val, 4),
                'std': round(std_val, 4),
                'median': round(float(np.median(values)), 4),
                'n': n,
            }

            if include_ci and n >= 2:
                # Normal approximation CI: mean +/- z * (std / sqrt(n))
                from scipy.stats import norm as normal_dist
                z = normal_dist.ppf(1 - (1 - ci_level) / 2)
                se = std_val / np.sqrt(n)
                entry['ci95_normal'] = [
                    round(mean_val - z * se, 4),
                    round(mean_val + z * se, 4),
                ]

                # Bootstrap CI
                ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap,
                                            ci=ci_level, seed=42)
                entry['ci95_boot'] = [round(ci_lo, 4), round(ci_hi, 4)]

            if include_per_sample:
                entry['per_sample'] = [round(float(v), 4) for v in values]

            agg[key] = entry
    return agg
