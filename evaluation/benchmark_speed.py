#!/usr/bin/env python
"""
Computational Efficiency Benchmark for All Methods.

Measures inference speed, GPU memory, tokens/second, and FLOPs for:
1. GECToR (Scheme C) - single-stage tagging
2. FELIX (Scheme D) - two-stage Tagger+Inserter
3. LevT (Scheme A, editing_v2) - iterative del/ins/tok
4. LLaMA AR (Scheme C) - autoregressive generation
5. BERT-CMLM (Scheme C) - iterative mask-predict
6. Discrete Diffusion (Scheme D) - SDEdit denoising (r03/r05/r07)

Usage:
    python evaluation/benchmark_speed.py
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# ==================== Paths ====================

# BeatEdit repo root (this file lives in <root>/evaluation/)
UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIOUS_DIR = os.path.dirname(UNIFIED_DIR)
sys.path.insert(0, UNIFIED_DIR)

# Test data paths: single-sample token dumps used to time inference.
# Generate them with the perturbation pipeline (see README, "Reproducing
# Table: efficiency"), or point these at your own dumps.
TEST_DATA_DIR = os.environ.get(
    "BEATEDIT_BENCH_DATA", os.path.join(UNIFIED_DIR, 'test_data', 'editing'))
TEST_DATA_C = os.path.join(TEST_DATA_DIR, 'C', '000.json')
TEST_DATA_D = os.path.join(TEST_DATA_DIR, 'D', '000.json')
TEST_DATA_A = os.path.join(TEST_DATA_DIR, 'A', '000.json')

# Output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'benchmark_results.json')

# Settings
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
LLAMA_MAX_TOKENS = 500  # Limit AR generation for tractability
DEVICE = 'cuda'

torch.backends.cudnn.benchmark = True


def estimate_flops_bert(hidden_size, num_layers, ffn_size, seq_len, vocab_size, num_passes=1):
    """
    Estimate FLOPs for a BERT-style transformer encoder forward pass.

    Per layer per position:
      - Self-attention: 4 * H^2 (Q,K,V projections + output projection) * 2 (multiply-add)
      - Attention scores: 2 * H * S (dot products)
      - FFN: 2 * H * F * 2 (two linear layers, multiply-add)

    Returns GFLOPs (1e9).
    """
    # Per position per layer FLOPs (multiply-add = 2 ops)
    attn_flops = 8 * hidden_size * hidden_size  # QKV + output projections
    attn_score = 4 * hidden_size * seq_len  # attention dot products + weighted sum
    ffn_flops = 4 * hidden_size * ffn_size  # two linear layers
    per_layer_per_pos = attn_flops + attn_score + ffn_flops

    # Total
    total = per_layer_per_pos * num_layers * seq_len
    # Embedding + output projection
    total += 2 * vocab_size * hidden_size * seq_len
    # Scale by number of passes
    total *= num_passes

    return total / 1e9  # GFLOPs


def estimate_flops_llama(hidden_size, num_layers, ffn_size, num_tokens, vocab_size):
    """
    Estimate FLOPs for LLaMA autoregressive generation of num_tokens.

    With KV cache, each token only does attention over all past tokens.
    Approximation: average context length = num_tokens / 2.
    """
    avg_ctx = num_tokens / 2
    attn_flops = 8 * hidden_size * hidden_size  # QKV + output
    attn_score = 4 * hidden_size * avg_ctx  # attention with KV cache
    ffn_flops = 4 * hidden_size * ffn_size

    per_layer_per_token = attn_flops + attn_score + ffn_flops
    total = per_layer_per_token * num_layers * num_tokens
    # Embedding + LM head
    total += 2 * vocab_size * hidden_size * num_tokens

    return total / 1e9


def load_test_data(path):
    """Load a test sample JSON."""
    with open(path) as f:
        return json.load(f)


def measure_gpu_memory():
    """Get peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_memory():
    """Reset GPU memory stats."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def benchmark_fn(fn, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS, label=""):
    """Benchmark a function: warmup, then time `runs` iterations.

    Returns dict with timing stats and peak GPU memory.
    """
    # Warmup
    print(f"    Warming up ({warmup} runs)...")
    for i in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Reset memory tracking after warmup
    reset_gpu_memory()

    # Benchmark
    times = []
    print(f"    Benchmarking ({runs} runs)...")
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    peak_mem = measure_gpu_memory()

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'peak_gpu_mb': float(peak_mem),
        'runs': runs,
    }


# ==================== 1. GECToR ====================

def benchmark_gector():
    """Benchmark GECToR (Scheme C) inference."""
    print("\n" + "="*60)
    print("  1. GECToR (Scheme C)")
    print("="*60)

    from scheme_utils import SchemeLoader

    data = load_test_data(TEST_DATA_C)
    source_tokens = list(data['source_tokens'])
    if len(source_tokens) > 2048:
        source_tokens = source_tokens[:2048]

    loader = SchemeLoader('C')
    print("  Loading GECToR model...")
    model = loader.gector.load_model(device=DEVICE)

    input_len = len(source_tokens)
    print(f"  Input length: {input_len} tokens")

    def run_once():
        corrected, info = loader.gector.inference_single(
            model, list(source_tokens), device=DEVICE
        )
        return corrected

    stats = benchmark_fn(run_once, label="GECToR")
    stats['input_tokens'] = input_len

    # Effective tokens/second
    stats['tokens_per_sec'] = input_len / (stats['mean_ms'] / 1000)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M")

    del model
    torch.cuda.empty_cache()
    return stats


# ==================== 2. FELIX ====================

def benchmark_felix():
    """Benchmark FELIX (Scheme D) inference."""
    print("\n" + "="*60)
    print("  2. FELIX (Scheme D)")
    print("="*60)

    from scheme_utils import SchemeLoader

    data = load_test_data(TEST_DATA_D)
    source_tokens = list(data['source_tokens'])
    if len(source_tokens) > 2048:
        source_tokens = source_tokens[:2048]

    loader = SchemeLoader('D')
    print("  Loading FELIX pipeline...")
    pipeline = loader.felix.load_pipeline(device=DEVICE)

    input_len = len(source_tokens)
    print(f"  Input length: {input_len} tokens")

    def run_once():
        return loader.felix.run_pipeline(pipeline, list(source_tokens), num_iterations=2)

    stats = benchmark_fn(run_once, label="FELIX")
    stats['input_tokens'] = input_len

    stats['tokens_per_sec'] = input_len / (stats['mean_ms'] / 1000)

    # Count parameters (tagger + inserter)
    n_tagger = sum(p.numel() for p in pipeline.tagger.parameters())
    n_inserter = sum(p.numel() for p in pipeline.inserter.parameters())
    n_params = n_tagger + n_inserter
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6
    stats['tagger_params_m'] = n_tagger / 1e6
    stats['inserter_params_m'] = n_inserter / 1e6

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M (Tagger: {stats['tagger_params_m']:.1f}M + Inserter: {stats['inserter_params_m']:.1f}M)")

    del pipeline
    torch.cuda.empty_cache()
    return stats


# ==================== 3. LevT ====================

def benchmark_levt():
    """Benchmark LevT (Scheme A, editing_v2) inference."""
    # NOTE (release): this benchmark depends on components that are not part of
    # this code release (the standalone LevT runner (run_levt.py) and its checkpoints). Pre-computed timings for every method are in
    # results/benchmark_results.json (paper Table on efficiency).
    print("  SKIPPED: requires the standalone LevT runner (run_levt.py) and its checkpoints (not included in this release); "
          "see results/benchmark_results.json for the paper timings.")
    return None

    print("\n" + "="*60)
    print("  3. LevT (Scheme A, editing_v2)")
    print("="*60)

    LEVT_DIR = os.path.join(PREVIOUS_DIR, 'LevT_inpainting')
    sys.path.insert(0, LEVT_DIR)

    from scheme_utils import SchemeLoader

    # Import LevT-specific modules
    sys.path.insert(0, os.path.join(PREVIOUS_DIR, 'evaluation'))

    # Use run_levt functions
    from run_levt import load_levt_pipeline, run_editing_sample

    data = load_test_data(TEST_DATA_A)
    source_tokens = list(data['source_tokens'])
    if len(source_tokens) > 2048:
        source_tokens = source_tokens[:2048]

    loader = SchemeLoader('A')
    ckpt_path = os.path.join(
        PREVIOUS_DIR, 'LevT_training_results', 'editing_v2', 'scheme_a', 'levt_best.pt'
    )
    print(f"  Loading LevT model from {ckpt_path}...")

    # Fix PyTorch 2.7 fastpath NaN bug
    torch.backends.mha.set_fastpath_enabled(False)

    pipeline = load_levt_pipeline(ckpt_path, device=DEVICE)

    input_len = len(source_tokens)
    print(f"  Input length: {input_len} tokens")

    def run_once():
        return run_editing_sample(
            pipeline, loader, data, device=DEVICE,
            all_accomp_editable=False, del_threshold=0.5
        )

    stats = benchmark_fn(run_once, label="LevT")
    stats['input_tokens'] = input_len

    stats['tokens_per_sec'] = input_len / (stats['mean_ms'] / 1000)

    n_params = sum(p.numel() for p in pipeline.model.parameters())
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M")

    del pipeline
    torch.cuda.empty_cache()
    return stats


# ==================== 4. LLaMA AR ====================

def benchmark_llama():
    # NOTE (release): this benchmark depends on components that are not part of
    # this code release (the PianoLLaMA baseline generator (model.py/config.py, external to this repo)). Pre-computed timings for every method are in
    # results/benchmark_results.json (paper Table on efficiency).
    print("  SKIPPED: requires the PianoLLaMA baseline generator (model.py/config.py, external to this repo) (not included in this release); "
          "see results/benchmark_results.json for the paper timings.")
    return None

    """Benchmark LLaMA AR (Scheme C) inference."""
    print("\n" + "="*60)
    print("  4. LLaMA AR (Scheme C)")
    print("="*60)

    llama_dir = os.path.join(PREVIOUS_DIR, 'encoding', 'with_pair')
    sys.path.insert(0, llama_dir)

    from config import ModelConfig
    from model import PianoLLaMA
    from transformers import LlamaConfig
    import safetensors.torch

    model_config = ModelConfig()

    # Build LLaMA config
    token_config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        rope_theta=model_config.rope_theta,
        attention_dropout=model_config.dropout,
        use_cache=True,
        initializer_range=0.02,
    )

    model_path = os.path.join(
        PREVIOUS_DIR, 'encoding', 'with_pair',
        'checkpoints_continue', 'epoch_2_0211_2022', 'model.safetensors'
    )
    print(f"  Loading LLaMA model from {model_path}...")

    model = PianoLLaMA(token_config).to(DEVICE)
    weights = safetensors.torch.load_file(model_path)
    model.load_state_dict(weights, strict=True)
    model.eval()

    # Use the test data to create a prompt (first 3 tokens: BOS + time_sig + bpm)
    data = load_test_data(TEST_DATA_C)
    source_tokens = data['source_tokens']
    # Extract prompt from source: BOS + time_sig + bpm (first 3 tokens)
    prompt_tokens = source_tokens[:3]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

    print(f"  Prompt length: {len(prompt_tokens)} tokens")
    print(f"  Max generation: {LLAMA_MAX_TOKENS} tokens")

    # Track per-token time
    per_token_times = []

    @torch.no_grad()
    def run_once():
        """Generate LLAMA_MAX_TOKENS tokens autoregressively."""
        generated = prompt.clone()
        past_key_values = None

        for step in range(LLAMA_MAX_TOKENS):
            outputs = model.model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Greedy decoding for deterministic timing
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == model_config.eos_token_id:
                break

        return generated

    # Warmup
    print(f"    Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        run_once()
    torch.cuda.synchronize()

    reset_gpu_memory()

    # Benchmark total time
    times = []
    output_lens = []
    print(f"    Benchmarking ({BENCHMARK_RUNS} runs)...")
    for i in range(BENCHMARK_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        output_lens.append(result.shape[1] - len(prompt_tokens))

    peak_mem = measure_gpu_memory()
    times = np.array(times)
    avg_output_len = np.mean(output_lens)

    # Also measure per-token latency with a single detailed run
    print("    Measuring per-token latency...")
    torch.cuda.synchronize()
    generated = prompt.clone()
    past_key_values = None
    per_token_times = []

    with torch.no_grad():
        for step in range(min(100, LLAMA_MAX_TOKENS)):  # Measure first 100 tokens
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model.model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_token_times.append((t1 - t0) * 1000)

            if next_token.item() == model_config.eos_token_id:
                break

    per_token_times = np.array(per_token_times)

    stats = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'peak_gpu_mb': float(peak_mem),
        'runs': BENCHMARK_RUNS,
        'input_tokens': len(prompt_tokens),
        'avg_output_tokens': float(avg_output_len),
        'max_gen_tokens': LLAMA_MAX_TOKENS,
        'per_token_mean_ms': float(np.mean(per_token_times[1:])) if len(per_token_times) > 1 else float(per_token_times[0]),
        'per_token_std_ms': float(np.std(per_token_times[1:])) if len(per_token_times) > 1 else 0.0,
        'first_token_ms': float(per_token_times[0]),
        'tokens_per_sec': float(avg_output_len / (np.mean(times) / 1000)),
    }

    n_params = sum(p.numel() for p in model.parameters())
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6

    # Extrapolate to full sequence length (~1700 tokens for scheme C)
    full_seq_len = len(data['source_tokens'])
    estimated_full_ms = stats['first_token_ms'] + stats['per_token_mean_ms'] * (full_seq_len - len(prompt_tokens))
    stats['estimated_full_seq_ms'] = estimated_full_ms
    stats['estimated_full_seq_tokens'] = full_seq_len

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms ({LLAMA_MAX_TOKENS} tokens)")
    print(f"  Per-token: {stats['per_token_mean_ms']:.2f} +/- {stats['per_token_std_ms']:.2f} ms")
    print(f"  First token: {stats['first_token_ms']:.2f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M")
    print(f"  Estimated full seq ({full_seq_len} tokens): {estimated_full_ms:.0f} ms")

    del model
    torch.cuda.empty_cache()

    # Remove llama_dir from path to avoid conflicts
    if llama_dir in sys.path:
        sys.path.remove(llama_dir)

    return stats


# ==================== 5. BERT-CMLM ====================

def benchmark_cmlm():
    # NOTE (release): this benchmark depends on components that are not part of
    # this code release (run_baselines.py helpers (external to this repo)). Pre-computed timings for every method are in
    # results/benchmark_results.json (paper Table on efficiency).
    print("  SKIPPED: requires run_baselines.py helpers (external to this repo) (not included in this release); "
          "see results/benchmark_results.json for the paper timings.")
    return None

    """Benchmark BERT-CMLM (Scheme C) inference."""
    print("\n" + "="*60)
    print("  5. BERT-CMLM (Scheme C)")
    print("="*60)

    from scheme_utils import SchemeLoader, SCHEME_INFO
    from safetensors.torch import load_file
    from transformers import BertConfig, BertForMaskedLM

    data = load_test_data(TEST_DATA_C)
    source_tokens = list(data['source_tokens'])
    if len(source_tokens) > 2048:
        source_tokens = source_tokens[:2048]

    info = SCHEME_INFO['C']
    loader = SchemeLoader('C')

    # Load BERT
    print("  Loading BERT for CMLM (Scheme C)...")
    bert_config = BertConfig(
        vocab_size=info['vocab_size'], hidden_size=512, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=2048, max_position_embeddings=2048,
        pad_token_id=info['pad_token_id'], type_vocab_size=1,
    )
    model = BertForMaskedLM(bert_config)
    state_dict = load_file(info['bert_checkpoint'])
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(DEVICE)

    # Get accompaniment positions (same as run_baselines.py)
    from run_baselines import get_accomp_note_positions
    accomp_pos = get_accomp_note_positions(
        source_tokens, loader.parse_sequence, loader.separate_tracks,
        info['note_min'], info['note_max'], info['has_marker'],
    )

    input_len = len(source_tokens)
    mask_token_id = info['mask_token_id']
    print(f"  Input length: {input_len} tokens, accomp positions: {len(accomp_pos)}")

    @torch.no_grad()
    def run_once():
        """CMLM: 1 iteration of mask-predict."""
        seq = list(source_tokens)
        n_accomp = len(accomp_pos)
        if n_accomp == 0:
            return seq

        # Iteration 1: mask 50% of accomp, predict
        ratio = 0.5
        num_mask = max(1, int(n_accomp * ratio))
        ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        attn = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=attn).logits[0]
        probs = torch.softmax(logits, dim=-1)
        confs = [(p, probs[p, seq[p]].item()) for p in accomp_pos]
        confs.sort(key=lambda x: x[1])
        mask_pos = [p for p, _ in confs[:num_mask]]

        for p in mask_pos:
            seq[p] = mask_token_id
        ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        attn = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=attn).logits[0]
        for p in mask_pos:
            seq[p] = logits[p].argmax().item()

        return seq

    stats = benchmark_fn(run_once, label="BERT-CMLM")
    stats['input_tokens'] = input_len
    stats['accomp_positions'] = len(accomp_pos)

    stats['tokens_per_sec'] = input_len / (stats['mean_ms'] / 1000)

    n_params = sum(p.numel() for p in model.parameters())
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M")

    del model
    torch.cuda.empty_cache()
    return stats


# ==================== Main ====================

# ==================== 6. Discrete Diffusion ====================

def benchmark_diffusion(start_ratio=0.3):
    # NOTE (release): this benchmark depends on components that are not part of
    # this code release (the discrete-diffusion baseline package (external to this repo)). Pre-computed timings for every method are in
    # results/benchmark_results.json (paper Table on efficiency).
    print("  SKIPPED: requires the discrete-diffusion baseline package (external to this repo) (not included in this release); "
          "see results/benchmark_results.json for the paper timings.")
    return None

    """Benchmark Discrete Diffusion SDEdit (Scheme D) inference."""
    ratio_label = f"r{int(start_ratio*10):02d}"
    print("\n" + "="*60)
    print(f"  6. Discrete Diffusion SDEdit ({ratio_label}, Scheme D)")
    print("="*60)

    diff_dir = os.path.join(PREVIOUS_DIR, 'Advanced_Experiments', 'discrete_diffusion')
    sys.path.insert(0, os.path.join(PREVIOUS_DIR))

    from Advanced_Experiments.discrete_diffusion.inference import DiffusionInference

    data = load_test_data(TEST_DATA_D)
    source_tokens = list(data['source_tokens'])
    if len(source_tokens) > 2048:
        source_tokens = source_tokens[:2048]

    ckpt_path = os.path.join(diff_dir, 'checkpoints', 'scheme_d', 'best_model.pt')
    print(f"  Loading Diffusion model from {ckpt_path}...")

    infer = DiffusionInference(
        scheme='D', checkpoint_path=ckpt_path, device=DEVICE,
        T=100, schedule='cosine'
    )

    input_len = len(source_tokens)
    num_steps = max(1, int(100 * start_ratio))
    print(f"  Input length: {input_len} tokens")
    print(f"  Denoising steps: {num_steps}")

    def run_once():
        return infer.sdedit_argmax(source_tokens, start_ratio=start_ratio)

    stats = benchmark_fn(run_once, label=f"Diffusion {ratio_label}")
    stats['input_tokens'] = input_len
    stats['denoising_steps'] = num_steps
    stats['start_ratio'] = start_ratio

    stats['tokens_per_sec'] = input_len / (stats['mean_ms'] / 1000)

    n_params = sum(p.numel() for p in infer.model.parameters())
    stats['num_params'] = n_params
    stats['num_params_m'] = n_params / 1e6

    print(f"  Result: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  Peak GPU: {stats['peak_gpu_mb']:.0f} MB")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Parameters: {stats['num_params_m']:.1f}M")

    del infer
    torch.cuda.empty_cache()
    return stats


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "="*90)
    print("  INFERENCE SPEED BENCHMARK SUMMARY")
    print("="*90)

    # Header
    print(f"{'Method':<25} {'Scheme':<8} {'Time (ms)':<20} {'GPU (MB)':<12} {'Tok/sec':<12} {'Params (M)':<12} {'Speedup':<8}")
    print("-"*90)

    # Find LLaMA time for speedup calculation
    # Use estimated full sequence time for fair comparison
    llama_time = results.get('llama', {}).get('estimated_full_seq_ms',
                 results.get('llama', {}).get('mean_ms', 1.0))

    for key, label, scheme in [
        ('gector', 'GECToR', 'C'),
        ('felix', 'FELIX', 'D'),
        ('levt', 'LevT (editing_v2)', 'A'),
        ('cmlm', 'BERT-CMLM', 'C'),
        ('diffusion_r03', 'Diffusion r03', 'D'),
        ('diffusion_r05', 'Diffusion r05', 'D'),
        ('diffusion_r07', 'Diffusion r07', 'D'),
        ('llama', 'LLaMA AR', 'C'),
    ]:
        if key not in results:
            continue
        r = results[key]
        mean = r['mean_ms']
        std = r['std_ms']
        gpu = r['peak_gpu_mb']
        tps = r['tokens_per_sec']
        params = r['num_params_m']

        # For LLaMA, use estimated full sequence time for speedup
        if key == 'llama':
            compare_time = r.get('estimated_full_seq_ms', mean)
        else:
            compare_time = mean

        speedup = llama_time / compare_time if compare_time > 0 else 0

        time_str = f"{mean:.1f} +/- {std:.1f}"
        print(f"{label:<25} {scheme:<8} {time_str:<20} {gpu:<12.0f} {tps:<12.0f} {params:<12.1f} {speedup:<8.1f}x")

    # Extra LLaMA details
    if 'llama' in results:
        r = results['llama']
        print("\n  LLaMA AR Details:")
        print(f"    Per-token latency: {r['per_token_mean_ms']:.2f} +/- {r['per_token_std_ms']:.2f} ms")
        print(f"    First token latency: {r['first_token_ms']:.2f} ms")
        print(f"    Measured with {LLAMA_MAX_TOKENS} token limit, avg output: {r['avg_output_tokens']:.0f} tokens")
        if 'estimated_full_seq_ms' in r:
            print(f"    Estimated full seq ({r['estimated_full_seq_tokens']} tokens): {r['estimated_full_seq_ms']:.0f} ms")

    print("="*90)

    # Speedup table (relative to LLaMA full-sequence estimate)
    print("\n  Speedup Ratios (relative to LLaMA AR estimated full-sequence time):")
    print(f"  LLaMA AR estimated full-sequence: {llama_time:.0f} ms")
    for key, label in [('gector', 'GECToR'), ('felix', 'FELIX'), ('levt', 'LevT'), ('cmlm', 'BERT-CMLM'),
                        ('diffusion_r03', 'Diff r03'), ('diffusion_r05', 'Diff r05'), ('diffusion_r07', 'Diff r07')]:
        if key in results:
            compare_time = results[key]['mean_ms']
            speedup = llama_time / compare_time if compare_time > 0 else 0
            print(f"    {label}: {speedup:.1f}x faster")


def main():
    print("="*60)
    print("  Inference Speed Benchmark")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Warmup runs: {WARMUP_RUNS}")
    print(f"  Benchmark runs: {BENCHMARK_RUNS}")
    print(f"  LLaMA max tokens: {LLAMA_MAX_TOKENS}")
    print("="*60)

    results = OrderedDict()

    # 1. GECToR
    try:
        results['gector'] = benchmark_gector()
    except Exception as e:
        print(f"  GECToR FAILED: {e}")
        import traceback; traceback.print_exc()

    # 2. FELIX
    try:
        results['felix'] = benchmark_felix()
    except Exception as e:
        print(f"  FELIX FAILED: {e}")
        import traceback; traceback.print_exc()

    # 3. LevT
    try:
        results['levt'] = benchmark_levt()
    except Exception as e:
        print(f"  LevT FAILED: {e}")
        import traceback; traceback.print_exc()

    # 4. LLaMA AR
    try:
        results['llama'] = benchmark_llama()
    except Exception as e:
        print(f"  LLaMA FAILED: {e}")
        import traceback; traceback.print_exc()

    # 5. BERT-CMLM
    try:
        results['cmlm'] = benchmark_cmlm()
    except Exception as e:
        print(f"  BERT-CMLM FAILED: {e}")
        import traceback; traceback.print_exc()

    # 6. Discrete Diffusion (3 ratios)
    for ratio in [0.3, 0.5, 0.7]:
        key = f"diffusion_r{int(ratio*10):02d}"
        try:
            results[key] = benchmark_diffusion(start_ratio=ratio)
        except Exception as e:
            print(f"  Diffusion {ratio} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ==================== FLOPs Estimation ====================
    print("\n" + "="*60)
    print("  FLOPs Estimation")
    print("="*60)

    # Common sequence length for comparison
    SEQ_LEN = 1749  # typical Scheme C/D sequence length

    # BERT-512-8L architecture (shared by GECToR, FELIX, CMLM, Diffusion, LevT)
    BERT_H, BERT_L, BERT_F = 512, 8, 2048

    flops_table = {}

    # GECToR: ~2 forward passes (max_iterations=2 for C/D)
    gector_passes = 2
    gector_flops = estimate_flops_bert(BERT_H, BERT_L, BERT_F, SEQ_LEN, 7145, gector_passes)
    flops_table['gector'] = {'gflops': round(gector_flops, 1), 'forward_passes': gector_passes, 'note': 'max 2 iters, usually converges in 1-2'}

    # FELIX: 2 iterations × (1 Tagger + 1 Inserter) = 4 forward passes
    felix_passes = 4
    felix_flops = estimate_flops_bert(BERT_H, BERT_L, BERT_F, SEQ_LEN, 7145, felix_passes)
    flops_table['felix'] = {'gflops': round(felix_flops, 1), 'forward_passes': felix_passes, 'note': '2 iters × (Tagger + Inserter)'}

    # LevT: ~3-5 iterations × 3 heads = 9-15 forward passes
    levt_passes_min, levt_passes_max = 9, 15
    levt_flops_avg = estimate_flops_bert(BERT_H, BERT_L, BERT_F, SEQ_LEN, 7146, 12)  # avg 12
    flops_table['levt'] = {'gflops': round(levt_flops_avg, 1), 'forward_passes': '9-15', 'note': '3-5 iters × 3 heads (del/ins/tok)'}

    # CMLM: 2 forward passes (1 scoring + 1 prediction)
    cmlm_passes = 2
    cmlm_flops = estimate_flops_bert(BERT_H, BERT_L, BERT_F, SEQ_LEN, 7145, cmlm_passes)
    flops_table['cmlm'] = {'gflops': round(cmlm_flops, 1), 'forward_passes': cmlm_passes, 'note': '1 scoring + 1 mask-predict'}

    # Diffusion: denoising_steps forward passes
    for ratio, key in [(0.3, 'diffusion_r03'), (0.5, 'diffusion_r05'), (0.7, 'diffusion_r07')]:
        steps = max(1, int(100 * ratio))
        diff_flops = estimate_flops_bert(BERT_H, BERT_L, BERT_F, SEQ_LEN, 7145, steps)
        flops_table[key] = {'gflops': round(diff_flops, 1), 'forward_passes': steps, 'note': f'{steps} denoising steps (T=100, ratio={ratio})'}

    # LLaMA: autoregressive, ~1749 forward passes with KV cache
    LLAMA_H, LLAMA_L, LLAMA_F, LLAMA_V = 768, 16, 3072, 7144
    llama_flops = estimate_flops_llama(LLAMA_H, LLAMA_L, LLAMA_F, SEQ_LEN, LLAMA_V)
    flops_table['llama'] = {'gflops': round(llama_flops, 1), 'forward_passes': SEQ_LEN, 'note': f'{SEQ_LEN} autoregressive steps with KV cache'}

    # Add FLOPs to results
    for key in flops_table:
        if key in results:
            results[key]['estimated_gflops'] = flops_table[key]['gflops']
            results[key]['forward_passes'] = flops_table[key]['forward_passes']
            results[key]['flops_note'] = flops_table[key]['note']

    # Print FLOPs table
    print(f"\n  {'Method':<25s} {'Fwd Passes':<14s} {'GFLOPs':<12s} {'Note'}")
    print(f"  {'-'*80}")
    for key in ['gector', 'felix', 'levt', 'cmlm', 'diffusion_r03', 'diffusion_r05', 'diffusion_r07', 'llama']:
        if key in flops_table:
            f = flops_table[key]
            print(f"  {key:<25s} {str(f['forward_passes']):<14s} {f['gflops']:<12.1f} {f['note']}")

    # Print summary
    results = OrderedDict((k, v) for k, v in results.items() if v is not None)
    print_summary_table(results)

    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
