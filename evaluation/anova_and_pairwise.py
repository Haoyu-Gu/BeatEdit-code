#!/usr/bin/env python
"""
Two-way ANOVA (Method × Encoding) + pairwise comparisons.

Usage:
    conda run --no-capture-output -n musictoken python -u \
        unified_eval/anova_and_pairwise.py --task correction
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import combinations
from collections import defaultdict

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UNIFIED_DIR)

from metrics import compute_all_metrics
from scheme_utils import SchemeLoader


# ==================== Load per-sample scores ====================

def load_per_sample_scores(task, method, scheme, scope='perturbed_only', metric='beat_exact_match'):
    """Load per-sample scores from result JSON or recompute from test_data + predictions."""
    result_path = os.path.join(UNIFIED_DIR, 'results', f'{task}_{method}_{scheme}_{scope}.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        if 'per_sample' in d:
            return [s[metric] for s in d['per_sample'] if metric in s]

    # Recompute from test_data + predictions
    test_dir = os.path.join(UNIFIED_DIR, 'test_data', task, scheme)
    pred_dir = os.path.join(UNIFIED_DIR, 'predictions', task, method, scheme)
    if not os.path.isdir(test_dir) or not os.path.isdir(pred_dir):
        return None

    loader = SchemeLoader(scheme)
    scores = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith('.json'):
            continue
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            continue

        with open(os.path.join(test_dir, fname)) as f:
            test = json.load(f)
        with open(pred_path) as f:
            pred = json.load(f)

        target_tokens = test['target_tokens']
        pred_tokens = pred.get('corrected_tokens', pred.get('predicted_tokens', pred.get('tokens', [])))
        changed = test.get('changed_beat_indices', [])

        if scope == 'perturbed_only' and changed:
            metrics = compute_all_metrics(pred_tokens, target_tokens, loader, beat_indices=changed)
        else:
            metrics = compute_all_metrics(pred_tokens, target_tokens, loader)

        if metric in metrics:
            scores.append(metrics[metric])

    return scores if scores else None


# ==================== Two-way ANOVA ====================

def two_way_anova(data_matrix, method_names, scheme_names):
    """
    Two-way ANOVA (Method × Encoding) on per-sample means.
    data_matrix: dict of (method, scheme) -> list of per-sample scores
    Returns F, p, eta² for each factor and interaction.
    """
    from scipy import stats

    methods = method_names
    schemes = scheme_names
    n_methods = len(methods)
    n_schemes = len(schemes)

    # Cell means and counts
    cell_means = {}
    cell_ns = {}
    all_scores = []
    for m in methods:
        for s in schemes:
            key = (m, s)
            scores = data_matrix.get(key, [])
            if not scores:
                print(f"  WARNING: no data for {m}×{s}")
                cell_means[key] = 0
                cell_ns[key] = 0
            else:
                cell_means[key] = np.mean(scores)
                cell_ns[key] = len(scores)
                all_scores.extend(scores)

    grand_mean = np.mean(all_scores) if all_scores else 0
    N = len(all_scores)

    # Method marginal means
    method_means = {}
    for m in methods:
        vals = []
        for s in schemes:
            vals.extend(data_matrix.get((m, s), []))
        method_means[m] = np.mean(vals) if vals else 0

    # Scheme marginal means
    scheme_means = {}
    for s in schemes:
        vals = []
        for m in methods:
            vals.extend(data_matrix.get((m, s), []))
        scheme_means[s] = np.mean(vals) if vals else 0

    # Sum of squares
    # SS_method
    ss_method = 0
    for m in methods:
        n_m = sum(cell_ns.get((m, s), 0) for s in schemes)
        ss_method += n_m * (method_means[m] - grand_mean) ** 2

    # SS_scheme
    ss_scheme = 0
    for s in schemes:
        n_s = sum(cell_ns.get((m, s), 0) for m in methods)
        ss_scheme += n_s * (scheme_means[s] - grand_mean) ** 2

    # SS_interaction
    ss_interaction = 0
    for m in methods:
        for s in schemes:
            n_ms = cell_ns.get((m, s), 0)
            if n_ms > 0:
                ss_interaction += n_ms * (cell_means[(m, s)] - method_means[m] - scheme_means[s] + grand_mean) ** 2

    # SS_total and SS_error
    ss_total = sum((x - grand_mean) ** 2 for x in all_scores)
    ss_error = ss_total - ss_method - ss_scheme - ss_interaction

    # Degrees of freedom
    df_method = n_methods - 1
    df_scheme = n_schemes - 1
    df_interaction = df_method * df_scheme
    df_error = N - n_methods * n_schemes
    df_total = N - 1

    # Mean squares and F
    results = {}
    for name, ss, df in [('Method', ss_method, df_method),
                          ('Encoding', ss_scheme, df_scheme),
                          ('Method×Encoding', ss_interaction, df_interaction)]:
        ms = ss / df if df > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1
        F = ms / ms_error if ms_error > 0 else 0
        p = 1 - stats.f.cdf(F, df, df_error) if df > 0 and df_error > 0 else 1
        eta_sq = ss / ss_total if ss_total > 0 else 0
        results[name] = {
            'SS': round(float(ss), 4),
            'df': int(df),
            'MS': round(float(ms), 6),
            'F': round(float(F), 2),
            'p': float(p),
            'eta_sq': round(float(eta_sq), 4),
        }

    results['Error'] = {
        'SS': round(float(ss_error), 4),
        'df': int(df_error),
        'MS': round(float(ss_error / df_error if df_error > 0 else 0), 6),
    }

    return results


# ==================== Pairwise tests ====================

def paired_bootstrap(scores_a, scores_b, n_bootstrap=10000, seed=42):
    scores_a = np.array(scores_a, dtype=np.float64)
    scores_b = np.array(scores_b, dtype=np.float64)
    n = len(scores_a)
    diffs = scores_a - scores_b
    observed = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot[i] = np.mean(diffs[idx])

    centered = boot - observed
    p = float(np.mean(np.abs(centered) >= abs(observed)))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    std_diff = float(np.std(diffs))
    cohens_d = observed / std_diff if std_diff > 0 else 0

    return {
        'delta': round(observed, 4),
        'ci95': (round(ci_lo, 4), round(ci_hi, 4)),
        'p_bootstrap': round(p, 4),
        'cohens_d': round(cohens_d, 3),
    }


def wilcoxon_test(scores_a, scores_b):
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(scores_a, scores_b)
        return round(float(p), 6)
    except Exception:
        return None


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['correction', 'editing', 'inpainting'])
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods for ANOVA')
    parser.add_argument('--schemes', type=str, default='A,B,C,D')
    parser.add_argument('--scope', type=str, default='perturbed_only')
    parser.add_argument('--metric', type=str, default='beat_exact_match')
    parser.add_argument('--pairwise', type=str, default=None,
                        help='Pairwise comparisons, e.g. "gector_B:gector_C,levt_editing_v2_A:gector_B"')
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(',')]

    # Default methods per task
    if args.methods is None:
        if args.task == 'correction':
            methods = ['gector', 'felix', 'levt_editing_v2']
        elif args.task == 'editing':
            methods = ['gector', 'felix', 'no_edit']
        else:
            methods = ['felix', 'levt_editing_v2', 'no_edit']
    else:
        methods = [m.strip() for m in args.methods.split(',')]

    print(f"Task: {args.task}, Metric: {args.metric}, Scope: {args.scope}")
    print(f"Methods: {methods}")
    print(f"Schemes: {schemes}")

    # Load all data
    data_matrix = {}
    print(f"\n{'Method':20s} ", end='')
    for s in schemes:
        print(f"  {s:>8s}", end='')
    print()
    print("-" * (22 + 10 * len(schemes)))

    for m in methods:
        print(f"{m:20s} ", end='')
        for s in schemes:
            scores = load_per_sample_scores(args.task, m, s, args.scope, args.metric)
            if scores:
                data_matrix[(m, s)] = scores
                print(f"  {np.mean(scores):>8.4f}", end='')
            else:
                print(f"  {'N/A':>8s}", end='')
        print()

    # Two-way ANOVA
    print(f"\n{'='*60}")
    print("  Two-way ANOVA (Method × Encoding)")
    print(f"{'='*60}")

    anova = two_way_anova(data_matrix, methods, schemes)
    print(f"\n{'Source':20s} {'SS':>10s} {'df':>4s} {'MS':>12s} {'F':>8s} {'p':>10s} {'η²':>8s}")
    print("-" * 74)
    for source in ['Method', 'Encoding', 'Method×Encoding', 'Error']:
        r = anova[source]
        if 'F' in r:
            p_str = f"{r['p']:.2e}" if r['p'] < 0.001 else f"{r['p']:.4f}"
            print(f"{source:20s} {r['SS']:>10.2f} {r['df']:>4d} {r['MS']:>12.6f} {r['F']:>8.2f} {p_str:>10s} {r['eta_sq']:>8.4f}")
        else:
            print(f"{source:20s} {r['SS']:>10.2f} {r['df']:>4d} {r['MS']:>12.6f}")

    # LaTeX
    print(f"\n% LaTeX ANOVA table rows:")
    for source in ['Method', 'Encoding', 'Method×Encoding']:
        r = anova[source]
        p_str = f"$<$0.001" if r['p'] < 0.001 else f"{r['p']:.3f}"
        print(f"  {source} & {r['df']} & {r['SS']:.2f} & {r['F']:.2f} & {p_str} & {r['eta_sq']:.4f} \\\\")

    # Pairwise comparisons
    if args.pairwise:
        print(f"\n{'='*60}")
        print("  Pairwise Comparisons")
        print(f"{'='*60}")

        pairs = args.pairwise.split(',')
        for pair in pairs:
            a_str, b_str = pair.strip().split(':')
            # Parse method_scheme
            parts_a = a_str.rsplit('_', 1)
            parts_b = b_str.rsplit('_', 1)
            method_a, scheme_a = parts_a[0], parts_a[1]
            method_b, scheme_b = parts_b[0], parts_b[1]

            scores_a = data_matrix.get((method_a, scheme_a))
            scores_b = data_matrix.get((method_b, scheme_b))

            if scores_a is None or scores_b is None:
                print(f"\n  {a_str} vs {b_str}: MISSING DATA")
                continue

            # Align sample counts
            n = min(len(scores_a), len(scores_b))
            scores_a = scores_a[:n]
            scores_b = scores_b[:n]

            boot = paired_bootstrap(scores_a, scores_b)
            wilcox_p = wilcoxon_test(scores_a, scores_b)

            print(f"\n  {a_str} vs {b_str} (n={n})")
            print(f"    Mean A: {np.mean(scores_a):.4f}, Mean B: {np.mean(scores_b):.4f}")
            print(f"    Δ = {boot['delta']:+.4f}, 95% CI = [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}]")
            print(f"    Bootstrap p = {boot['p_bootstrap']:.4f}, Wilcoxon p = {wilcox_p}")
            print(f"    Cohen's d = {boot['cohens_d']:.3f}")

            # LaTeX
            p_str = f"$<$0.001" if boot['p_bootstrap'] < 0.001 else f"{boot['p_bootstrap']:.3f}"
            print(f"    % LaTeX: {a_str} vs {b_str} & {boot['delta']:+.4f} & [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}] & {p_str} & {boot['cohens_d']:.3f} \\\\")

    # Save results
    output = {
        'task': args.task,
        'metric': args.metric,
        'scope': args.scope,
        'methods': methods,
        'schemes': schemes,
        'cell_means': {f"{m}_{s}": round(float(np.mean(data_matrix[(m, s)])), 4)
                       for m, s in data_matrix},
        'anova': anova,
    }
    out_path = os.path.join(UNIFIED_DIR, 'results', f'{args.task}_anova_{args.scope}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
