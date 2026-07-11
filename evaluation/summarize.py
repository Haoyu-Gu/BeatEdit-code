#!/usr/bin/env python
"""
Summarize evaluation results into paper-ready tables.

Reads results/{task}_*_{scope}.json -> generates Markdown tables for the paper.

Usage:
    conda run --no-capture-output -n musictoken python -u \
        unified_eval/summarize.py --task editing
        unified_eval/summarize.py --task correction
        unified_eval/summarize.py --task inpainting
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(results_dir, scope='perturbed_only', task='editing'):
    """Load all result JSON files for a given task."""
    pattern = os.path.join(results_dir, f'{task}_*_{scope}.json')
    files = sorted(glob.glob(pattern))

    results = []
    for fpath in files:
        basename = os.path.basename(fpath)
        # Skip combined file and significance test results
        if 'all_results' in basename or 'significance' in basename:
            continue
        with open(fpath) as f:
            data = json.load(f)
        # Sanity check: must have 'method' key (skip non-standard files)
        if 'method' not in data:
            continue
        results.append(data)
    return results


def build_lookup(results):
    """Build (method, scheme) -> result dict."""
    lookup = {}
    for r in results:
        lookup[(r['method'], r['scheme'])] = r
    return lookup


def format_val(val, metric):
    """Format a metric value for display."""
    if val is None:
        return '---'
    if metric == 'mean_pitch_error':
        return f'{val:.2f}'
    if metric == 'fmd':
        return f'{val:.2f}'
    return f'{val:.3f}'


def format_val_ci(result, metric, show_ci=True):
    """Format a metric value with optional CI for display."""
    if result is None:
        return '---'
    if metric == 'fmd':
        val = result.get('fmd')
        if val is None:
            return '---'
        return f'{val:.2f}'
    overall = result.get('overall', {})
    if metric not in overall:
        return '---'
    entry = overall[metric]
    val = entry.get('mean')
    if val is None:
        return '---'

    base = format_val(val, metric)
    if not show_ci:
        return base

    # Prefer bootstrap CI
    ci = entry.get('ci95_boot')
    if ci is not None:
        if metric == 'mean_pitch_error':
            return f'{base} [{ci[0]:.2f},{ci[1]:.2f}]'
        else:
            return f'{base} [{ci[0]:.3f},{ci[1]:.3f}]'
    return base


def get_mean(result, metric):
    """Extract mean value from a result dict."""
    if result is None:
        return None
    if metric == 'fmd':
        return result.get('fmd')
    overall = result.get('overall', {})
    if metric in overall:
        return overall[metric].get('mean')
    return None


def get_n(result, metric):
    """Extract sample count from a result dict."""
    if result is None:
        return None
    overall = result.get('overall', {})
    if metric in overall:
        return overall[metric].get('n')
    return result.get('num_samples')


# ==================== Table Generators ====================

def table_main(lookup, schemes, methods, metrics, show_ci=False):
    """
    Paper main table: method × scheme for each metric.
    Format: one sub-table per metric.
    """
    lines = []
    lines.append("## Main Results\n")

    for metric in metrics:
        lines.append(f"### {metric}\n")

        # Header
        header = "| Scheme |"
        sep = "|:------|"
        for m in methods:
            header += f" {m} |"
            sep += ":---:|"
        lines.append(header)
        lines.append(sep)

        # Rows
        for scheme in schemes:
            row = f"| {scheme} |"
            values = []
            for m in methods:
                r = lookup.get((m, scheme))
                val = get_mean(r, metric)
                values.append(val)

            # Find best value
            is_lower_better = metric in ('mean_pitch_error', 'fmd')
            valid_vals = [(v, i) for i, v in enumerate(values) if v is not None]
            best_idx = None
            if valid_vals:
                if is_lower_better:
                    best_idx = min(valid_vals, key=lambda x: x[0])[1]
                else:
                    best_idx = max(valid_vals, key=lambda x: x[0])[1]

            for i, (val, m) in enumerate(zip(values, methods)):
                r = lookup.get((m, scheme))
                formatted = format_val_ci(r, metric, show_ci=show_ci)
                if i == best_idx and val is not None:
                    row += f" **{formatted}** |"
                else:
                    row += f" {formatted} |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def table_encoding_ranking(lookup, schemes, methods, metric='beat_exact_match'):
    """Encoding ranking table: for each method, show scheme ranking."""
    lines = []
    lines.append(f"## Encoding Rankings ({metric})\n")

    header = "| Method |"
    sep = "|:------|"
    for rank in range(1, len(schemes) + 1):
        header += f" Rank {rank} |"
        sep += ":---:|"
    lines.append(header)
    lines.append(sep)

    is_lower_better = metric in ('mean_pitch_error', 'fmd')

    for method in methods:
        row = f"| {method} |"
        vals = []
        for scheme in schemes:
            r = lookup.get((method, scheme))
            val = get_mean(r, metric)
            if val is not None:
                vals.append((val, scheme))

        # Sort
        vals.sort(key=lambda x: x[0], reverse=not is_lower_better)

        for val, scheme in vals:
            row += f" {scheme}({format_val(val, metric)}) |"
        # Pad if fewer schemes
        for _ in range(len(schemes) - len(vals)):
            row += " --- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def table_per_level(lookup, schemes, methods, metric='beat_exact_match'):
    """Per-level breakdown table."""
    lines = []
    lines.append(f"## Per-Level Breakdown ({metric})\n")

    levels = ['L1', 'L2', 'L3', 'L4']

    for method in methods:
        lines.append(f"### {method}\n")

        header = "| Scheme |"
        sep = "|:------|"
        for lv in levels:
            header += f" {lv} |"
            sep += ":---:|"
        lines.append(header)
        lines.append(sep)

        for scheme in schemes:
            row = f"| {scheme} |"
            r = lookup.get((method, scheme))
            if r is None:
                for _ in levels:
                    row += " --- |"
            else:
                per_level = r.get('per_level', {})
                for lv in levels:
                    lv_data = per_level.get(lv, {})
                    lv_metrics = lv_data.get('metrics', {})
                    if metric in lv_metrics:
                        val = lv_metrics[metric].get('mean')
                        row += f" {format_val(val, metric)} |"
                    else:
                        row += " --- |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def table_improvement_over_noedit(lookup, schemes, methods_to_compare,
                                   metric='beat_exact_match'):
    """Improvement over No-Edit baseline."""
    lines = []
    lines.append(f"## Improvement over No-Edit ({metric})\n")

    header = "| Scheme |"
    sep = "|:------|"
    for m in methods_to_compare:
        header += f" {m} |"
        sep += ":---:|"
    lines.append(header)
    lines.append(sep)

    is_lower_better = metric in ('mean_pitch_error', 'fmd')

    for scheme in schemes:
        row = f"| {scheme} |"
        noedit_r = lookup.get(('no_edit', scheme))
        noedit_val = get_mean(noedit_r, metric)

        for m in methods_to_compare:
            r = lookup.get((m, scheme))
            val = get_mean(r, metric)
            if val is not None and noedit_val is not None:
                diff = val - noedit_val
                if is_lower_better:
                    diff = -diff  # positive = improvement for lower-is-better
                row += f" {diff:+.4f} |"
            else:
                row += " --- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Summarize results')
    parser.add_argument('--task', type=str, default='editing',
                        choices=['editing', 'correction', 'inpainting'],
                        help='Task type to summarize')
    parser.add_argument('--scope', type=str, default='perturbed_only',
                        choices=['perturbed_only', 'full_sequence'])
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(UNIFIED_DIR, 'results'))
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: results/SUMMARY_{task}_{scope}.md)')
    parser.add_argument('--show_ci', action='store_true',
                        help='Show 95%% bootstrap CI in main table')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, f'SUMMARY_{args.task}_{args.scope}.md')

    results = load_results(args.results_dir, args.scope, args.task)
    if not results:
        print(f"No results found in {args.results_dir} for task={args.task}, scope={args.scope}")
        return

    lookup = build_lookup(results)

    # Discover available schemes and methods
    schemes = sorted(set(r['scheme'] for r in results))
    methods_set = set(r['method'] for r in results)
    # Canonical order
    method_order = ['no_edit', 'copy_ctx', 'cmlm', 'felix', 'gector',
                     'levt_inpainting', 'levt_editing', 'levt_track_aware',
                     'levt_editing_tb', 'levt_editing_full', 'levt_vanilla_edit',
                     'levt_vanilla_t03', 'levt_editing_t03', 'levt_track_aware_edit']
    methods = [m for m in method_order if m in methods_set]

    our_methods = [m for m in methods if m not in ('no_edit', 'copy_ctx', 'cmlm')]

    print(f"Summarizing: {len(results)} results")
    print(f"  Task: {args.task}")
    print(f"  Schemes: {schemes}")
    print(f"  Methods: {methods}")
    print(f"  Scope: {args.scope}")

    # Generate tables
    output_lines = [
        f"# Unified Evaluation Summary — {args.task} ({args.scope})\n",
        f"Generated from unified_eval framework.\n",
    ]

    # Main table with key metrics
    main_metrics = ['beat_exact_match', 'note_f1_tol0', 'note_f1_tol2',
                     'chroma_f1', 'mean_pitch_error']
    if any(r.get('fmd') is not None for r in results):
        main_metrics.append('fmd')

    output_lines.append(table_main(lookup, schemes, methods, main_metrics,
                                    show_ci=args.show_ci))
    output_lines.append(table_encoding_ranking(lookup, schemes, methods))

    if our_methods:
        output_lines.append(table_improvement_over_noedit(
            lookup, schemes, our_methods))

    output_lines.append(table_per_level(lookup, schemes, methods))

    # Write output
    output_text = "\n".join(output_lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(output_text)

    print(f"\nSummary written to {args.output}")

    # Also print to stdout
    print(f"\n{'='*80}")
    print(output_text)


if __name__ == '__main__':
    main()
