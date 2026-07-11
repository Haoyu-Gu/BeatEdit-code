#!/usr/bin/env python
"""
Statistical Significance Tests for Unified Evaluation.

Performs paired comparisons between methods using:
  1. Paired Bootstrap Test (paired_bootstrap_p)
  2. Wilcoxon Signed-Rank Test (scipy.stats.wilcoxon)

Reads per-sample scores from result JSONs (requires --per_sample flag during evaluate.py).
If per-sample scores are not available, re-computes from test_data + predictions.

Usage:
    # Compare two methods on one task:
    conda run --no-capture-output -n musictoken python -u \
        unified_eval/statistical_tests.py --task editing \
        --method_a gector --method_b felix --schemes A,B,C,D

    # Full pairwise comparison table:
    conda run --no-capture-output -n musictoken python -u \
        unified_eval/statistical_tests.py --task editing --all_pairs \
        --methods gector,felix,no_edit --schemes A,B,C,D

    # Use existing result JSONs with per_sample data:
    conda run --no-capture-output -n musictoken python -u \
        unified_eval/statistical_tests.py --task editing \
        --method_a gector --method_b felix --from_results
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import combinations

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UNIFIED_DIR)


# ==================== Statistical Tests ====================

def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap significance test.

    Tests H0: mean(scores_a) == mean(scores_b)
    Returns p-value (two-sided).

    Args:
        scores_a: array of per-sample scores for method A
        scores_b: array of per-sample scores for method B (same samples, aligned)
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with keys: p_value, observed_diff, ci95_diff
    """
    scores_a = np.array(scores_a, dtype=np.float64)
    scores_b = np.array(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b), \
        f"Sample sizes must match: {len(scores_a)} vs {len(scores_b)}"

    n = len(scores_a)
    diffs = scores_a - scores_b
    observed_diff = float(np.mean(diffs))

    rng = np.random.RandomState(seed)

    # Bootstrap: resample diffs, count how often bootstrap mean crosses zero
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[sample_idx])

    # Two-sided p-value: fraction of bootstrap diffs on opposite side of zero
    # (or more extreme) relative to observed
    # Standard approach: test if zero is within the bootstrap distribution
    centered_boot = boot_diffs - observed_diff  # center around zero
    p_value = float(np.mean(np.abs(centered_boot) >= abs(observed_diff)))

    # CI of the difference
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        'p_value': round(p_value, 4),
        'observed_diff': round(observed_diff, 4),
        'ci95_diff': [round(ci_lower, 4), round(ci_upper, 4)],
        'n_samples': n,
        'n_bootstrap': n_bootstrap,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


def wilcoxon_test(scores_a, scores_b):
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    Returns dict with p_value and statistic.
    """
    from scipy.stats import wilcoxon

    scores_a = np.array(scores_a, dtype=np.float64)
    scores_b = np.array(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b)

    diffs = scores_a - scores_b
    # If all diffs are zero, p=1.0
    if np.all(diffs == 0):
        return {
            'p_value': 1.0,
            'statistic': 0.0,
            'significant_005': False,
            'significant_001': False,
        }

    try:
        stat, p = wilcoxon(scores_a, scores_b, alternative='two-sided')
        return {
            'p_value': round(float(p), 6),
            'statistic': round(float(stat), 4),
            'significant_005': float(p) < 0.05,
            'significant_001': float(p) < 0.01,
        }
    except Exception as e:
        return {
            'p_value': float('nan'),
            'statistic': float('nan'),
            'error': str(e),
            'significant_005': False,
            'significant_001': False,
        }


# ==================== Per-sample score extraction ====================

def load_per_sample_from_results(results_dir, task, method, scheme, scope='perturbed_only'):
    """
    Load per-sample scores from a result JSON that was generated with --per_sample.

    Returns:
        dict of metric_name -> list of per-sample values, or None if not available
    """
    fpath = os.path.join(results_dir, f'{task}_{method}_{scheme}_{scope}.json')
    if not os.path.exists(fpath):
        return None

    with open(fpath) as f:
        data = json.load(f)

    overall = data.get('overall', {})
    per_sample = {}
    for key, entry in overall.items():
        if 'per_sample' in entry:
            per_sample[key] = entry['per_sample']

    if not per_sample:
        return None
    return per_sample


def compute_per_sample_scores(task, method, scheme, scope='perturbed_only',
                               test_data_dir=None, predictions_dir=None):
    """
    Recompute per-sample scores from test data and predictions.
    This is a fallback when --per_sample was not used during evaluate.py.
    """
    import glob as glob_mod
    from scheme_utils import SchemeLoader
    from metrics import compute_all_metrics

    if test_data_dir is None:
        test_data_dir = os.path.join(UNIFIED_DIR, 'test_data', task)
    if predictions_dir is None:
        predictions_dir = os.path.join(UNIFIED_DIR, 'predictions', task)

    test_dir = os.path.join(test_data_dir, scheme)
    pred_dir = os.path.join(predictions_dir, method, scheme)

    if not os.path.exists(pred_dir):
        print(f"  No predictions for {method}/{scheme}")
        return None

    loader = SchemeLoader(scheme)
    decode_fn = loader.decode_beat

    test_files = sorted(glob_mod.glob(os.path.join(test_dir, '*.json')))
    per_sample_metrics = []

    for fpath in test_files:
        fname = os.path.basename(fpath)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            continue

        with open(fpath) as f:
            test_data = json.load(f)
        with open(pred_path) as f:
            pred_data = json.load(f)

        target_tokens = test_data['target_tokens']
        pred_tokens = pred_data['pred_tokens']
        changed_indices = set(test_data['changed_beat_indices'])

        try:
            tgt_parsed = loader.parse_sequence(target_tokens)
            pred_parsed = loader.parse_sequence(pred_tokens)
            _, tgt_accomp = loader.separate_tracks(tgt_parsed)
            _, pred_accomp = loader.separate_tracks(pred_parsed)

            if scope == 'perturbed_only' and len(changed_indices) > 0:
                n = min(len(pred_accomp), len(tgt_accomp))
                pred_filtered = [pred_accomp[i] for i in range(n) if i in changed_indices]
                tgt_filtered = [tgt_accomp[i] for i in range(n) if i in changed_indices]
                changed_mask = [True] * len(pred_filtered)
            else:
                pred_filtered = pred_accomp
                tgt_filtered = tgt_accomp
                changed_mask = [i in changed_indices for i in range(min(len(pred_accomp), len(tgt_accomp)))]

            m = compute_all_metrics(pred_filtered, tgt_filtered, decode_fn,
                                    changed_mask=changed_mask if scope == 'full_sequence' else None)
            per_sample_metrics.append(m)

        except Exception as e:
            continue

    if not per_sample_metrics:
        return None

    # Convert list-of-dicts to dict-of-lists
    result = {}
    for key in per_sample_metrics[0].keys():
        values = [m.get(key) for m in per_sample_metrics
                  if m.get(key) is not None and np.isfinite(m.get(key, float('nan')))]
        if values:
            result[key] = values

    return result


def get_per_sample_scores(task, method, scheme, scope='perturbed_only',
                           results_dir=None, test_data_dir=None, predictions_dir=None,
                           from_results=False):
    """
    Get per-sample scores, either from saved results or by recomputing.
    """
    if results_dir is None:
        results_dir = os.path.join(UNIFIED_DIR, 'results')

    if from_results:
        scores = load_per_sample_from_results(results_dir, task, method, scheme, scope)
        if scores is not None:
            return scores
        print(f"  Warning: no per_sample data in results for {method}/{scheme}, recomputing...")

    return compute_per_sample_scores(task, method, scheme, scope,
                                      test_data_dir, predictions_dir)


# ==================== Comparison ====================

def compare_methods(task, method_a, method_b, schemes, scope='perturbed_only',
                    metrics=None, n_bootstrap=10000, from_results=False,
                    results_dir=None):
    """
    Compare two methods across all schemes.

    Returns:
        dict of scheme -> metric -> test_results
    """
    if metrics is None:
        metrics = ['beat_exact_match', 'note_f1_tol0', 'mean_pitch_error', 'chroma_f1']

    all_comparisons = {}

    for scheme in schemes:
        print(f"\n  Scheme {scheme}: {method_a} vs {method_b}")

        scores_a = get_per_sample_scores(
            task, method_a, scheme, scope,
            from_results=from_results, results_dir=results_dir)
        scores_b = get_per_sample_scores(
            task, method_b, scheme, scope,
            from_results=from_results, results_dir=results_dir)

        if scores_a is None or scores_b is None:
            print(f"    SKIP: missing data for {method_a if scores_a is None else method_b}")
            continue

        scheme_results = {}
        for metric in metrics:
            if metric not in scores_a or metric not in scores_b:
                continue

            sa = scores_a[metric]
            sb = scores_b[metric]

            # Ensure same length (should be if same test set)
            n = min(len(sa), len(sb))
            if n < 2:
                continue
            sa = sa[:n]
            sb = sb[:n]

            # Run tests
            boot_result = paired_bootstrap_test(sa, sb, n_bootstrap=n_bootstrap)
            wilcox_result = wilcoxon_test(sa, sb)

            scheme_results[metric] = {
                'method_a_mean': round(float(np.mean(sa)), 4),
                'method_b_mean': round(float(np.mean(sb)), 4),
                'paired_bootstrap': boot_result,
                'wilcoxon': wilcox_result,
            }

            sig = "*" if boot_result['significant_005'] else ""
            sig2 = "**" if boot_result['significant_001'] else ""
            marker = sig2 if sig2 else sig
            print(f"    {metric}: {method_a}={np.mean(sa):.4f} vs {method_b}={np.mean(sb):.4f} "
                  f"  diff={boot_result['observed_diff']:+.4f} "
                  f"  bootstrap p={boot_result['p_value']:.4f}{marker} "
                  f"  wilcoxon p={wilcox_result['p_value']:.4f}")

        all_comparisons[scheme] = scheme_results

    return all_comparisons


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Statistical significance tests')
    parser.add_argument('--task', type=str, default='editing',
                        choices=['editing', 'correction', 'inpainting'])
    parser.add_argument('--scope', type=str, default='perturbed_only',
                        choices=['perturbed_only', 'full_sequence'])
    parser.add_argument('--schemes', type=str, default='A,B,C,D')
    parser.add_argument('--method_a', type=str, default=None,
                        help='First method to compare')
    parser.add_argument('--method_b', type=str, default=None,
                        help='Second method to compare')
    parser.add_argument('--methods', type=str, default='no_edit,felix,gector',
                        help='Methods for --all_pairs mode')
    parser.add_argument('--all_pairs', action='store_true',
                        help='Compare all pairs of methods')
    parser.add_argument('--metrics', type=str,
                        default='beat_exact_match,note_f1_tol0,mean_pitch_error,chroma_f1',
                        help='Metrics to compare (comma-separated)')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap resamples (default: 10000)')
    parser.add_argument('--from_results', action='store_true',
                        help='Load per-sample scores from result JSONs (requires --per_sample during eval)')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(UNIFIED_DIR, 'results'))
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: results/{task}_significance_{scope}.json)')
    args = parser.parse_args()

    schemes = [s.strip().upper() for s in args.schemes.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]

    if args.output is None:
        args.output = os.path.join(args.results_dir,
                                    f'{args.task}_significance_{args.scope}.json')

    print(f"Statistical Significance Tests")
    print(f"  Task: {args.task}")
    print(f"  Scope: {args.scope}")
    print(f"  Schemes: {schemes}")
    print(f"  Metrics: {metrics}")
    print(f"  Bootstrap: {args.n_bootstrap}")

    all_results = {}

    if args.all_pairs:
        methods = [m.strip() for m in args.methods.split(',')]
        print(f"  All-pairs mode: {methods}")

        for ma, mb in combinations(methods, 2):
            key = f"{ma}_vs_{mb}"
            print(f"\n{'='*60}")
            print(f"  {ma} vs {mb}")
            print(f"{'='*60}")

            all_results[key] = compare_methods(
                args.task, ma, mb, schemes, args.scope,
                metrics, args.n_bootstrap, args.from_results,
                args.results_dir)

    elif args.method_a and args.method_b:
        key = f"{args.method_a}_vs_{args.method_b}"
        all_results[key] = compare_methods(
            args.task, args.method_a, args.method_b, schemes, args.scope,
            metrics, args.n_bootstrap, args.from_results,
            args.results_dir)

    else:
        print("ERROR: specify --method_a and --method_b, or use --all_pairs")
        sys.exit(1)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print_significance_summary(all_results, metrics)


def print_significance_summary(all_results, metrics):
    """Print a compact significance summary table."""
    print(f"\n{'='*80}")
    print(f"  SIGNIFICANCE SUMMARY")
    print(f"{'='*80}")

    for comparison, scheme_data in all_results.items():
        methods = comparison.split('_vs_')
        print(f"\n  {methods[0]} vs {methods[1]}:")

        for metric in metrics:
            print(f"\n    {metric}:")
            print(f"    {'Scheme':<8s} {'A_mean':>8s} {'B_mean':>8s} {'Diff':>8s} "
                  f"{'Boot_p':>8s} {'Wilc_p':>8s} {'Sig':>5s}")
            print(f"    {'-'*55}")

            for scheme, scheme_results in sorted(scheme_data.items()):
                if metric not in scheme_results:
                    continue
                mr = scheme_results[metric]
                boot_p = mr['paired_bootstrap']['p_value']
                wilc_p = mr['wilcoxon']['p_value']
                diff = mr['paired_bootstrap']['observed_diff']
                sig = "**" if boot_p < 0.01 else ("*" if boot_p < 0.05 else "")
                print(f"    {scheme:<8s} {mr['method_a_mean']:8.4f} {mr['method_b_mean']:8.4f} "
                      f"{diff:+8.4f} {boot_p:8.4f} {wilc_p:8.6f} {sig:>5s}")


if __name__ == '__main__':
    main()
