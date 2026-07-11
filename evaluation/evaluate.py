#!/usr/bin/env python
"""
Unified Evaluation: compute all metrics for all methods × all schemes × all tasks.

Reads:
    test_data/{task}/{scheme}/*.json  (target + metadata)
    predictions/{method}/{scheme}/*.json  (pred_tokens)

Outputs:
    results/{task}_{method}_{scheme}_{scope}.json  (per method-scheme)
    results/{task}_all_results_{scope}.json  (combined)

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n musictoken python -u \
        unified_eval/evaluate.py --task editing --schemes A,B,C,D --methods gector,felix,no_edit,copy_ctx,cmlm

    # Different tasks:
    unified_eval/evaluate.py --task correction
    unified_eval/evaluate.py --task inpainting

    # Perturbed-only scope (default, paper main table):
    unified_eval/evaluate.py --scope perturbed_only

    # Full sequence scope (appendix):
    unified_eval/evaluate.py --scope full_sequence

    # Include LevT variants:
    unified_eval/evaluate.py --task inpainting --methods levt_inpainting --schemes A,B,C,D
    unified_eval/evaluate.py --task editing --methods levt_editing --schemes C,D
    unified_eval/evaluate.py --task inpainting --methods levt_track_aware --schemes D
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from collections import defaultdict
from safetensors.torch import load_file
from transformers import BertConfig, BertModel

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UNIFIED_DIR)

from scheme_utils import SchemeLoader, SCHEME_INFO
from metrics import (
    beat_exact_match, note_f1, mean_pitch_error, chroma_f1,
    rhythm_similarity, token_accuracy, context_preservation,
    compute_all_metrics, compute_fmd, safe_aggregate,
    MAIN_METRIC_KEYS, ALL_METRIC_KEYS,
)


# ==================== BERT Embedding ====================

@torch.no_grad()
def get_bert_embedding(bert_model, tokens, device='cuda'):
    """Extract mean-pooled BERT embedding (512-dim)."""
    if len(tokens) > 2048:
        tokens = tokens[:2048]
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    outputs = bert_model(input_ids=ids, attention_mask=mask)
    hidden = outputs.last_hidden_state[0]  # (seq_len, 512)
    return hidden.mean(dim=0).cpu().numpy()  # (512,)


def load_bert_encoder(scheme, device='cuda'):
    """Load BERT encoder (without MLM head) for embeddings."""
    info = SCHEME_INFO[scheme]
    bert_config = BertConfig(
        vocab_size=info['vocab_size'], hidden_size=512, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=2048, max_position_embeddings=2048,
        pad_token_id=info['pad_token_id'], type_vocab_size=1,
    )
    model = BertModel(bert_config)

    # Load weights from MLM checkpoint (only take bert.* weights)
    state_dict = load_file(info['bert_checkpoint'])
    # Filter: remove 'cls.' prefix keys (MLM head)
    bert_state = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            bert_state[k[5:]] = v  # strip 'bert.' prefix
        elif not k.startswith('cls.'):
            bert_state[k] = v
    model.load_state_dict(bert_state, strict=False)
    model.eval().to(device)
    return model


# ==================== Evaluation Core ====================

def evaluate_method_scheme(method, scheme, test_data_dir, predictions_dir,
                           scope='perturbed_only', bert_model=None, device='cuda',
                           task='editing', include_per_sample=False,
                           include_ci=True, n_bootstrap=1000):
    """
    Evaluate one method on one scheme.

    Args:
        scope: 'perturbed_only' — only evaluate perturbed accompaniment beats
               'full_sequence' — evaluate all accompaniment beats
        task: 'editing', 'correction', or 'inpainting'
    """
    test_dir = os.path.join(test_data_dir, scheme)
    pred_dir = os.path.join(predictions_dir, method, scheme)

    if not os.path.exists(pred_dir):
        print(f"  No predictions for {method}/{scheme}")
        return None

    loader = SchemeLoader(scheme)
    decode_fn = loader.decode_beat

    test_files = sorted(glob.glob(os.path.join(test_dir, '*.json')))
    all_metrics = []
    level_metrics = defaultdict(list)
    pred_embeddings = []
    ref_embeddings = []
    errors = 0

    for fpath in test_files:
        fname = os.path.basename(fpath)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            errors += 1
            continue

        with open(fpath) as f:
            test_data = json.load(f)
        with open(pred_path) as f:
            pred_data = json.load(f)

        target_tokens = test_data['target_tokens']
        pred_tokens = pred_data['pred_tokens']
        level = test_data['level']
        changed_indices = set(test_data['changed_beat_indices'])

        try:
            # Parse and separate tracks
            tgt_parsed = loader.parse_sequence(target_tokens)
            pred_parsed = loader.parse_sequence(pred_tokens)
            _, tgt_accomp = loader.separate_tracks(tgt_parsed)
            _, pred_accomp = loader.separate_tracks(pred_parsed)

            if scope == 'perturbed_only' and len(changed_indices) > 0:
                # Filter to only perturbed beats
                n = min(len(pred_accomp), len(tgt_accomp))
                pred_filtered = [pred_accomp[i] for i in range(n) if i in changed_indices]
                tgt_filtered = [tgt_accomp[i] for i in range(n) if i in changed_indices]

                # Also build changed_mask for context_preservation (all True in filtered)
                changed_mask = [True] * len(pred_filtered)
            else:
                pred_filtered = pred_accomp
                tgt_filtered = tgt_accomp
                changed_mask = [i in changed_indices for i in range(min(len(pred_accomp), len(tgt_accomp)))]

            m = compute_all_metrics(pred_filtered, tgt_filtered, decode_fn,
                                    changed_mask=changed_mask if scope == 'full_sequence' else None)

            # Context preservation (only meaningful for full_sequence)
            if scope == 'full_sequence':
                cp_mask = [i not in changed_indices for i in range(min(len(pred_accomp), len(tgt_accomp)))]
                n_unchanged = sum(cp_mask)
                n_preserved = sum(1 for i in range(min(len(pred_accomp), len(tgt_accomp)))
                                  if cp_mask[i] and pred_accomp[i]['tokens'] == tgt_accomp[i]['tokens'])
                m['context_preservation'] = n_preserved / max(n_unchanged, 1)

            m['level'] = level
            all_metrics.append(m)
            level_metrics[level].append(m)

            # BERT embeddings for FMD
            if bert_model is not None:
                pred_emb = get_bert_embedding(bert_model, pred_tokens, device)
                ref_emb = get_bert_embedding(bert_model, target_tokens, device)
                pred_embeddings.append(pred_emb)
                ref_embeddings.append(ref_emb)

                cos_sim = float(np.dot(pred_emb, ref_emb) /
                               (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb) + 1e-8))
                m['bert_cosine_sim'] = cos_sim

        except Exception as e:
            print(f"  ERROR evaluating {fname}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    if len(all_metrics) == 0:
        print(f"  No valid samples for {method}/{scheme}")
        return None

    # Aggregate
    result = {
        'method': method,
        'scheme': scheme,
        'task': task,
        'scope': scope,
        'num_samples': len(all_metrics),
        'num_errors': errors,
        'overall': safe_aggregate(all_metrics, include_per_sample=include_per_sample,
                                  include_ci=include_ci, n_bootstrap=n_bootstrap),
    }

    # Per-level
    per_level = {}
    for lv in sorted(level_metrics.keys()):
        per_level[lv] = {
            'count': len(level_metrics[lv]),
            'metrics': safe_aggregate(level_metrics[lv],
                                      include_per_sample=include_per_sample,
                                      include_ci=include_ci,
                                      n_bootstrap=n_bootstrap),
        }
    result['per_level'] = per_level

    # FMD
    if len(pred_embeddings) >= 10:
        result['fmd'] = round(compute_fmd(pred_embeddings, ref_embeddings), 4)
    else:
        result['fmd'] = None

    return result


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Unified evaluation')
    parser.add_argument('--task', type=str, default='editing',
                        choices=['editing', 'correction', 'inpainting'],
                        help='Task type to evaluate')
    parser.add_argument('--schemes', type=str, default='A,B,C,D')
    parser.add_argument('--methods', type=str,
                        default='no_edit,copy_ctx,cmlm,felix,gector',
                        help='Methods to evaluate (comma-separated). '
                             'Available: no_edit,copy_ctx,cmlm,felix,gector,'
                             'levt_inpainting,levt_editing,levt_track_aware')
    parser.add_argument('--scope', type=str, default='perturbed_only',
                        choices=['perturbed_only', 'full_sequence'])
    parser.add_argument('--compute_fmd', action='store_true',
                        help='Compute FMD (requires GPU, slower)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Test data directory (default: test_data/{task})')
    parser.add_argument('--predictions_dir', type=str,
                        default=os.path.join(UNIFIED_DIR, 'predictions'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(UNIFIED_DIR, 'results'))
    parser.add_argument('--per_sample', action='store_true',
                        help='Include per-sample scores in output JSON')
    parser.add_argument('--no_ci', action='store_true',
                        help='Skip confidence interval computation')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap resamples for CI (default: 1000)')
    args = parser.parse_args()

    if args.test_data_dir is None:
        args.test_data_dir = os.path.join(UNIFIED_DIR, 'test_data', args.task)

    # Include task in predictions directory: predictions/{task}/{method}/{scheme}/
    args.predictions_dir = os.path.join(args.predictions_dir, args.task)

    schemes = [s.strip().upper() for s in args.schemes.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Unified Evaluation")
    print(f"  Task: {args.task}")
    print(f"  Schemes: {schemes}")
    print(f"  Methods: {methods}")
    print(f"  Scope: {args.scope}")
    print(f"  FMD: {args.compute_fmd}")

    all_results = []

    for scheme in schemes:
        print(f"\n{'='*60}")
        print(f"  Scheme {scheme}")
        print(f"{'='*60}")

        # Load BERT for FMD if requested
        bert_model = None
        if args.compute_fmd:
            print(f"  Loading BERT encoder for FMD...")
            bert_model = load_bert_encoder(scheme, device=args.device)

        for method in methods:
            # Check if predictions exist
            pred_dir = os.path.join(args.predictions_dir, method, scheme)
            if not os.path.exists(pred_dir) or len(os.listdir(pred_dir)) == 0:
                print(f"  Skipping {method}/{scheme} (no predictions)")
                continue

            print(f"\n  --- {method} ({args.task}) ---")
            result = evaluate_method_scheme(
                method, scheme, args.test_data_dir, args.predictions_dir,
                scope=args.scope, bert_model=bert_model, device=args.device,
                task=args.task, include_per_sample=args.per_sample,
                include_ci=not args.no_ci, n_bootstrap=args.n_bootstrap,
            )

            if result is not None:
                all_results.append(result)

                # Print summary
                overall = result['overall']
                print(f"    Samples: {result['num_samples']}, Errors: {result['num_errors']}")
                for key in MAIN_METRIC_KEYS:
                    if key in overall:
                        entry = overall[key]
                        ci_str = ""
                        if 'ci95_boot' in entry:
                            ci_str = f"  95%CI[{entry['ci95_boot'][0]:.4f}, {entry['ci95_boot'][1]:.4f}]"
                        print(f"    {key}: {entry['mean']:.4f} (±{entry['std']:.4f}, n={entry.get('n', '?')}){ci_str}")
                if result['fmd'] is not None:
                    print(f"    FMD: {result['fmd']:.4f}")

                # Save per-method-scheme result
                out_path = os.path.join(args.output_dir,
                                        f"{args.task}_{method}_{scheme}_{args.scope}.json")
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        # Cleanup BERT
        if bert_model is not None:
            del bert_model
            torch.cuda.empty_cache()

    # Save combined results
    combined_path = os.path.join(args.output_dir, f'{args.task}_all_results_{args.scope}.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {combined_path}")

    # Print cross-scheme summary table
    if len(all_results) > 1:
        print_summary_table(all_results, schemes, methods)


def print_summary_table(results, schemes, methods):
    """Print a cross-scheme summary table with confidence intervals."""
    print(f"\n{'='*100}")
    print(f"  SUMMARY TABLE (with 95% Bootstrap CI)")
    print(f"{'='*100}")

    # Build lookup: (method, scheme) -> result
    lookup = {}
    for r in results:
        lookup[(r['method'], r['scheme'])] = r

    for metric in ['beat_exact_match', 'note_f1_tol0', 'mean_pitch_error']:
        print(f"\n  {metric}:")
        header = f"    {'Scheme':<8s}"
        for m in methods:
            header += f" {m:>24s}"
        print(header)
        print(f"    {'-'*(8 + 25*len(methods))}")

        for scheme in schemes:
            row = f"    {scheme:<8s}"
            for m in methods:
                r = lookup.get((m, scheme))
                if r and metric in r['overall']:
                    entry = r['overall'][metric]
                    v = entry['mean']
                    if 'ci95_boot' in entry:
                        ci = entry['ci95_boot']
                        cell = f"{v:.4f} [{ci[0]:.3f},{ci[1]:.3f}]"
                    else:
                        cell = f"{v:.4f}"
                    row += f" {cell:>24s}"
                else:
                    row += f" {'---':>24s}"
            print(row)


if __name__ == '__main__':
    main()
