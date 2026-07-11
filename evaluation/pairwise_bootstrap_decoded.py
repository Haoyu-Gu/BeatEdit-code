#!/usr/bin/env python
"""
Pairwise bootstrap significance tests on decoded beat_exact_match.

Uses per-sample decoded beat_exact from reeval_decoded_beat_exact.py output,
NOT the token-space beat_exact_match from evaluate.py.

Usage:
    python \
        evaluation/pairwise_bootstrap_decoded.py
"""

import os
import sys
import json
import numpy as np
from scipy.stats import wilcoxon

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Data Loading ====================

def load_decoded_per_sample(reeval_full_path=None):
    """
    Load per-sample decoded beat_exact from reeval_decoded_beat_exact.py output.

    Returns:
        dict of (task, method, scheme) -> list of per-sample decoded beat_exact scores
    """
    if reeval_full_path is None:
        reeval_full_path = os.path.join(UNIFIED_DIR, 'results',
                                         'beat_exact_decoded_reeval_full.json')

    with open(reeval_full_path) as f:
        data = json.load(f)

    lookup = {}
    for r in data:
        key = (r['task'], r['method'], r['scheme'])
        lookup[key] = r['new_per_sample']  # decoded beat_exact per sample

    return lookup


# ==================== Statistical Tests ====================

def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap significance test.
    H0: mean(scores_a) == mean(scores_b)
    Returns: dict with delta, CI, p, cohens_d
    """
    scores_a = np.array(scores_a, dtype=np.float64)
    scores_b = np.array(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b), \
        f"Sample sizes must match: {len(scores_a)} vs {len(scores_b)}"

    n = len(scores_a)
    diffs = scores_a - scores_b
    observed = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot[i] = np.mean(diffs[idx])

    # Two-sided p-value
    centered = boot - observed
    p = float(np.mean(np.abs(centered) >= abs(observed)))

    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    # Cohen's d (paired)
    std_diff = float(np.std(diffs, ddof=1))
    cohens_d = observed / std_diff if std_diff > 1e-10 else 0.0

    return {
        'mean_a': round(float(np.mean(scores_a)), 4),
        'mean_b': round(float(np.mean(scores_b)), 4),
        'delta': round(observed, 4),
        'ci_low': round(ci_lo, 4),
        'ci_high': round(ci_hi, 4),
        'p': round(p, 4),
        'cohens_d': round(cohens_d, 3),
        'n': n,
        'n_bootstrap': n_bootstrap,
    }


def run_wilcoxon(scores_a, scores_b):
    """Wilcoxon signed-rank test (two-sided)."""
    diffs = np.array(scores_a) - np.array(scores_b)
    if np.all(diffs == 0):
        return 1.0
    try:
        stat, p = wilcoxon(scores_a, scores_b, alternative='two-sided')
        return round(float(p), 6)
    except Exception:
        return None


# ==================== Comparisons to Run ====================

# (label, task, method_a, scheme_a, method_b, scheme_b)
COMPARISONS = [
    ("IterEdit^A vs SeqTag^B Correction",
     "correction", "levt_editing_v2", "A", "gector", "B"),

    ("IterEdit vs SeqTag Editing A",
     "editing", "levt_editing_v2", "A", "gector", "A"),

    ("IterEdit vs SeqTag Editing D",
     "editing", "levt_editing_v2", "D", "gector", "D"),

    ("SeqTag vs TagFill Correction A",
     "correction", "gector", "A", "felix", "A"),

    ("SeqTag vs TagFill Correction D",
     "correction", "gector", "D", "felix", "D"),

    ("IterEdit vs TagFill Editing A",
     "editing", "levt_editing_v2", "A", "felix", "A"),

    ("IterEdit vs TagFill Editing D",
     "editing", "levt_editing_v2", "D", "felix", "D"),

    ("TagFill vs No-Edit Completion A",
     "inpainting", "felix", "A", "no_edit", "A"),

    ("TagFill vs No-Edit Completion D",
     "inpainting", "felix", "D", "no_edit", "D"),
]


# ==================== Main ====================

def main():
    print("=" * 100)
    print("  Pairwise Bootstrap (B=10000) — decoded beat_exact_match, perturbed_only")
    print("=" * 100)

    # Load per-sample data
    lookup = load_decoded_per_sample()
    print(f"  Loaded {len(lookup)} method×scheme×task entries")

    # Check availability
    for label, task, m_a, s_a, m_b, s_b in COMPARISONS:
        key_a = (task, m_a, s_a)
        key_b = (task, m_b, s_b)
        if key_a not in lookup:
            print(f"  WARNING: {key_a} not found!")
        if key_b not in lookup:
            print(f"  WARNING: {key_b} not found!")

    # Run comparisons
    results = []
    print(f"\n{'comparison':<42s} {'task':<12s} {'sch':>3s} {'mean_A':>7s} {'mean_B':>7s} "
          f"{'delta':>7s} {'CI_low':>7s} {'CI_hi':>7s} {'p':>7s} {'d':>6s} {'n':>4s}")
    print("-" * 110)

    for label, task, m_a, s_a, m_b, s_b in COMPARISONS:
        key_a = (task, m_a, s_a)
        key_b = (task, m_b, s_b)

        if key_a not in lookup or key_b not in lookup:
            print(f"  {label}: SKIPPED (missing data)")
            continue

        scores_a = lookup[key_a]
        scores_b = lookup[key_b]

        # Align sample counts
        n = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        # Bootstrap
        r = paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000)
        wilcox_p = run_wilcoxon(scores_a, scores_b)

        # Determine scheme label
        if s_a == s_b:
            sch_label = s_a
        else:
            sch_label = f"{s_a}/{s_b}"

        # Significance markers
        sig = ""
        if r['p'] < 0.001:
            sig = "***"
        elif r['p'] < 0.01:
            sig = "**"
        elif r['p'] < 0.05:
            sig = "*"

        print(f"{label:<42s} {task:<12s} {sch_label:>3s} "
              f"{r['mean_a']:>7.4f} {r['mean_b']:>7.4f} "
              f"{r['delta']:>+7.4f} {r['ci_low']:>7.4f} {r['ci_high']:>7.4f} "
              f"{r['p']:>7.4f} {r['cohens_d']:>+6.3f} {r['n']:>4d} {sig}")

        result_entry = {
            'comparison': label,
            'task': task,
            'method_a': m_a,
            'scheme_a': s_a,
            'method_b': m_b,
            'scheme_b': s_b,
            'mean_a': r['mean_a'],
            'mean_b': r['mean_b'],
            'delta': r['delta'],
            'ci_low': r['ci_low'],
            'ci_high': r['ci_high'],
            'p_bootstrap': r['p'],
            'cohens_d': r['cohens_d'],
            'p_wilcoxon': wilcox_p,
            'n': r['n'],
            'n_bootstrap': r['n_bootstrap'],
        }
        results.append(result_entry)

    # Print CSV-style output
    print(f"\n\n{'='*100}")
    print("  CSV output (for pasting into paper appendix)")
    print(f"{'='*100}")
    print("comparison,task,scheme,mean_A,mean_B,delta,CI_low,CI_high,p,cohens_d")
    for r in results:
        sch = r['scheme_a'] if r['scheme_a'] == r['scheme_b'] else f"{r['scheme_a']}/{r['scheme_b']}"
        print(f"{r['comparison']},{r['task']},{sch},"
              f"{r['mean_a']:.4f},{r['mean_b']:.4f},{r['delta']:+.4f},"
              f"{r['ci_low']:.4f},{r['ci_high']:.4f},{r['p_bootstrap']:.4f},{r['cohens_d']:.3f}")

    # Save JSON
    out_path = os.path.join(UNIFIED_DIR, 'results', 'pairwise_bootstrap_decoded.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Print LaTeX table rows
    print(f"\n\n{'='*100}")
    print("  LaTeX table rows")
    print(f"{'='*100}")
    for r in results:
        sch = r['scheme_a'] if r['scheme_a'] == r['scheme_b'] else f"{r['scheme_a']}/{r['scheme_b']}"
        p_str = "$<$0.001" if r['p_bootstrap'] < 0.001 else f"{r['p_bootstrap']:.3f}"
        sig = ""
        if r['p_bootstrap'] < 0.001:
            sig = "$^{***}$"
        elif r['p_bootstrap'] < 0.01:
            sig = "$^{**}$"
        elif r['p_bootstrap'] < 0.05:
            sig = "$^{*}$"
        print(f"  {r['comparison']} & {r['mean_a']:.3f} & {r['mean_b']:.3f} & "
              f"{r['delta']:+.3f} & [{r['ci_low']:.3f}, {r['ci_high']:.3f}] & "
              f"{p_str}{sig} & {r['cohens_d']:.2f} \\\\")


if __name__ == '__main__':
    main()
