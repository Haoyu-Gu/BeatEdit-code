#!/usr/bin/env python
"""
Re-evaluate beat_exact_match in decoded note space (cross-scheme comparable).

Old metric: compare raw token lists per beat (scheme-dependent)
New metric: decode tokens -> sorted list of (abs_pitch, patch_value), then compare

This makes the metric cross-scheme comparable since all schemes decode to
the same (pitch, value) space.

Usage:
    conda run --no-capture-output -n musictoken python -u unified_eval/reeval_decoded_beat_exact.py
"""

import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UNIFIED_DIR)

from scheme_utils import SchemeLoader

# ==================== Methods to evaluate ====================

# Paper Table 5a methods + baselines
TASK_METHODS = {
    'correction': [
        'no_edit', 'copy_ctx', 'cmlm',
        'gector', 'felix',
        'levt_editing_v2', 'levt_track_aware_v2',
        'levt_editing', 'levt_vanilla_edit',
        'diffusion', 'diffusion_r03', 'diffusion_r05', 'diffusion_r07',
        'llama_prompt', 'llama_full', 'llama_selective',
        'gector_s1only', 'gector_scratch',
        'levt_encdec',
    ],
    'editing': [
        'no_edit', 'copy_ctx', 'cmlm',
        'gector', 'felix',
        'levt_editing_v2', 'levt_track_aware_v2',
        'levt_editing', 'levt_vanilla_edit', 'levt_vanilla_t03',
        'diffusion', 'diffusion_r03', 'diffusion_r05', 'diffusion_r07',
        'llama_prompt', 'llama_teacher', 'llama_selective',
        'levt_encdec',
    ],
    'inpainting': [
        'no_edit', 'copy_ctx', 'cmlm',
        'felix',
        'levt_inpainting', 'levt_editing_v2_inp', 'levt_accomp_inp',
        'diffusion', 'diffusion_r03', 'diffusion_r05', 'diffusion_r07',
        'llama_prompt', 'llama_selective', 'llama_teacher', 'llama_detect_regen',
    ],
}

SCHEMES = ['A', 'B', 'C', 'D']


def decoded_beat_exact_match(pred_beats, tgt_beats, decode_fn):
    """
    Beat-level exact match in decoded note space.

    For each beat:
      1. Decode tokens -> list of (abs_pitch, patch_value)
      2. Sort the list
      3. Compare sorted lists

    Returns: fraction of beats that match exactly in note space.
    """
    n = min(len(pred_beats), len(tgt_beats))
    if n == 0:
        return 0.0
    matches = 0
    for i in range(n):
        pred_notes = sorted(decode_fn(pred_beats[i]['tokens']))
        tgt_notes = sorted(decode_fn(tgt_beats[i]['tokens']))
        if pred_notes == tgt_notes:
            matches += 1
    return matches / n


def token_beat_exact_match(pred_beats, tgt_beats):
    """Original token-space beat exact match (for comparison)."""
    n = min(len(pred_beats), len(tgt_beats))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if pred_beats[i]['tokens'] == tgt_beats[i]['tokens'])
    return matches / n


def evaluate_one(test_path, pred_path, loader, scope='perturbed_only', task='editing'):
    """Evaluate one sample, return both old and new beat_exact_match."""
    with open(test_path) as f:
        test_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    target_tokens = test_data['target_tokens']
    pred_tokens = pred_data['pred_tokens']
    changed_indices = set(test_data['changed_beat_indices'])

    tgt_parsed = loader.parse_sequence(target_tokens)
    pred_parsed = loader.parse_sequence(pred_tokens)
    _, tgt_accomp = loader.separate_tracks(tgt_parsed)
    _, pred_accomp = loader.separate_tracks(pred_parsed)

    if scope == 'perturbed_only' and len(changed_indices) > 0:
        n = min(len(pred_accomp), len(tgt_accomp))
        pred_filtered = []
        tgt_filtered = []
        for i in range(n):
            if i not in changed_indices:
                continue
            if task == 'inpainting' and len(tgt_accomp[i]['tokens']) == 0:
                continue  # target beat is empty — not a real perturbation
            pred_filtered.append(pred_accomp[i])
            tgt_filtered.append(tgt_accomp[i])
    else:
        pred_filtered = pred_accomp
        tgt_filtered = tgt_accomp

    decode_fn = loader.decode_beat
    old_score = token_beat_exact_match(pred_filtered, tgt_filtered)
    new_score = decoded_beat_exact_match(pred_filtered, tgt_filtered, decode_fn)

    return old_score, new_score


def evaluate_method_scheme(task, method, scheme, scope='perturbed_only'):
    """Evaluate all samples for one method/scheme/task combo."""
    test_dir = os.path.join(UNIFIED_DIR, 'test_data', task, scheme)
    pred_dir = os.path.join(UNIFIED_DIR, 'predictions', task, method, scheme)

    if not os.path.exists(pred_dir):
        return None
    pred_files = glob.glob(os.path.join(pred_dir, '*.json'))
    if len(pred_files) == 0:
        return None

    loader = SchemeLoader(scheme)

    old_scores = []
    new_scores = []
    errors = 0

    for pred_path in sorted(pred_files):
        fname = os.path.basename(pred_path)
        test_path = os.path.join(test_dir, fname)
        if not os.path.exists(test_path):
            errors += 1
            continue
        try:
            old_s, new_s = evaluate_one(test_path, pred_path, loader, scope, task=task)
            old_scores.append(old_s)
            new_scores.append(new_s)
        except Exception as e:
            errors += 1

    if len(old_scores) == 0:
        return None

    return {
        'task': task,
        'method': method,
        'scheme': scheme,
        'scope': scope,
        'n': len(old_scores),
        'errors': errors,
        'beat_exact_token': round(float(np.mean(old_scores)), 4),
        'beat_exact_decoded': round(float(np.mean(new_scores)), 4),
        'beat_exact_token_std': round(float(np.std(old_scores)), 4),
        'beat_exact_decoded_std': round(float(np.std(new_scores)), 4),
        'diff': round(float(np.mean(new_scores)) - float(np.mean(old_scores)), 4),
        'old_per_sample': [round(x, 4) for x in old_scores],
        'new_per_sample': [round(x, 4) for x in new_scores],
    }


def main():
    scope = 'perturbed_only'
    all_results = []

    for task in ['correction', 'editing', 'inpainting']:
        methods = TASK_METHODS[task]
        print(f"\n{'='*70}")
        print(f"  Task: {task}")
        print(f"{'='*70}")

        for method in methods:
            for scheme in SCHEMES:
                result = evaluate_method_scheme(task, method, scheme, scope)
                if result is not None:
                    all_results.append(result)
                    diff_str = f"{result['diff']:+.4f}" if result['diff'] != 0 else "  0"
                    print(f"  {method:25s} {scheme}  n={result['n']:3d}  "
                          f"token={result['beat_exact_token']:.4f}  "
                          f"decoded={result['beat_exact_decoded']:.4f}  "
                          f"diff={diff_str}")

    # Save full results
    out_path = os.path.join(UNIFIED_DIR, 'results', 'beat_exact_decoded_reeval.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save without per_sample for the summary
    summary_results = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ('old_per_sample', 'new_per_sample')}
        summary_results.append(s)

    with open(out_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save full results with per-sample data
    full_path = os.path.join(UNIFIED_DIR, 'results', 'beat_exact_decoded_reeval_full.json')
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results (with per-sample) saved to {full_path}")

    # Print summary tables
    print_tables(all_results)


def print_tables(all_results):
    """Print comparison tables."""
    # Build lookup
    lookup = {}
    for r in all_results:
        lookup[(r['task'], r['method'], r['scheme'])] = r

    # Paper main methods
    main_methods = {
        'correction': [('no_edit', 'No-Edit'), ('gector', 'SeqTag'), ('felix', 'TagFill'), ('levt_editing_v2', 'IterEdit')],
        'editing': [('no_edit', 'No-Edit'), ('gector', 'SeqTag'), ('felix', 'TagFill'), ('levt_editing_v2', 'IterEdit')],
        'inpainting': [('no_edit', 'No-Edit'), ('felix', 'TagFill'), ('levt_inpainting', 'IterEdit'), ('levt_inpainting_v2', 'IterEdit_v2')],
    }

    for task in ['correction', 'editing', 'inpainting']:
        print(f"\n{'='*80}")
        print(f"  {task.upper()} — beat_exact_match: Token vs Decoded")
        print(f"{'='*80}")
        print(f"  {'Method':<15s}  {'Metric':<8s}  {'A':>8s}  {'B':>8s}  {'C':>8s}  {'D':>8s}")
        print(f"  {'-'*60}")

        for method_key, method_label in main_methods[task]:
            # Token space
            row_tok = f"  {method_label:<15s}  {'token':<8s}"
            row_dec = f"  {'':<15s}  {'decoded':<8s}"
            for scheme in SCHEMES:
                r = lookup.get((task, method_key, scheme))
                if r:
                    row_tok += f"  {r['beat_exact_token']:>7.3f}"
                    row_dec += f"  {r['beat_exact_decoded']:>7.3f}"
                else:
                    row_tok += f"  {'---':>7s}"
                    row_dec += f"  {'---':>7s}"
            print(row_tok)
            print(row_dec)
            print()

    # No-Edit decoded comparison (the key diagnostic)
    print(f"\n{'='*80}")
    print(f"  NO-EDIT BASELINE: Token vs Decoded (key diagnostic)")
    print(f"{'='*80}")
    print(f"  {'Task':<15s}  {'Metric':<8s}  {'A':>8s}  {'B':>8s}  {'C':>8s}  {'D':>8s}")
    print(f"  {'-'*60}")
    for task in ['correction', 'editing', 'inpainting']:
        row_tok = f"  {task:<15s}  {'token':<8s}"
        row_dec = f"  {'':<15s}  {'decoded':<8s}"
        for scheme in SCHEMES:
            r = lookup.get((task, 'no_edit', scheme))
            if r:
                row_tok += f"  {r['beat_exact_token']:>7.3f}"
                row_dec += f"  {r['beat_exact_decoded']:>7.3f}"
            else:
                row_tok += f"  {'---':>7s}"
                row_dec += f"  {'---':>7s}"
        print(row_tok)
        print(row_dec)
        print()


if __name__ == '__main__':
    main()
