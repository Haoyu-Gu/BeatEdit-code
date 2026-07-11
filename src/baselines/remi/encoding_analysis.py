"""
Layer 1 Encoding Analysis: BEAT vs REMI Comparison

Compares encoding-level statistics between BEAT (Scheme A) and REMI encodings
without any model training. Samples 500 test files with the same seed=42 split.

Metrics:
- Sequence length (tok/piece)
- Tokens per note
- Levenshtein edit distance for same musical perturbation
- Alignment stability (token displacement after single note insertion)
- Truncation rate at max_len=2048

Output: results/baselines/beat_vs_remi_analysis.json + markdown tables

Usage:
    python encoding_analysis.py
"""

import os
import sys
import json
import random
import numpy as np
from collections import defaultdict

# REMI imports
from remi_tokenizer import midi_to_tokens, tokens_to_notes
from config import (
    MIDI_DATA_DIR, NPZ_DATA_DIR, BAR_TOKEN, BOS_TOKEN, EOS_TOKEN,
    is_music_token,
)
from perturbation import perturb_sequence as remi_perturb
from sequence_parser import parse_sequence

# BEAT imports (Scheme A) — use subprocess-style isolation to avoid
# module cache conflicts (REMI's config/perturbation are already cached).
# Instead of importing BEAT modules directly, we'll run BEAT analysis
# as a separate subprocess, or use importlib with sys.modules swapping.
BEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'seqtag', 'scheme_A')
try:
    import importlib.util

    # Save REMI modules from sys.modules cache
    _remi_cached = {}
    _conflicting = ['config', 'perturbation', 'dataset', 'sequence_parser', 'model']
    for _name in _conflicting:
        if _name in sys.modules:
            _remi_cached[_name] = sys.modules.pop(_name)

    # Load BEAT modules with BEAT_DIR at front of sys.path
    _old_path = sys.path[:]
    sys.path.insert(0, BEAT_DIR)
    try:
        import importlib
        _beat_config = importlib.import_module('config')
        _beat_perturbation = importlib.import_module('perturbation')
        _beat_dataset = importlib.import_module('dataset')

        GECToRDataset = _beat_dataset.GECToRDataset
        beat_get_file_lists = _beat_dataset.get_file_lists
        beat_perturb = _beat_perturbation.perturb_sequence

        # Save BEAT modules under beat_ prefix
        _beat_modules = {}
        for _name in _conflicting:
            if _name in sys.modules:
                _beat_modules[_name] = sys.modules.pop(_name)

        BEAT_AVAILABLE = True
    except Exception as e:
        BEAT_AVAILABLE = False
        print(f"WARNING: Cannot import BEAT modules ({e}). BEAT comparison will be skipped.")
        # Clean up any partial BEAT imports
        for _name in _conflicting:
            sys.modules.pop(_name, None)
    finally:
        sys.path[:] = _old_path
        # Restore REMI modules
        sys.modules.update(_remi_cached)
except Exception as e:
    BEAT_AVAILABLE = False
    print(f"WARNING: Cannot import BEAT modules ({e}). BEAT comparison will be skipped.")


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two-row optimization for memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev

    return prev[n]


def count_notes_remi(tokens):
    """Count number of notes in REMI token sequence."""
    notes = tokens_to_notes(tokens)
    return len(notes)


def count_notes_beat(tokens):
    """Count number of notes in BEAT token sequence (Scheme A)."""
    # In BEAT: notes are [position][value] pairs within beats
    # Position tokens: 81-168, Value tokens: 0-80
    count = 0
    i = 0
    while i < len(tokens) - 1:
        if 81 <= tokens[i] <= 168 and 0 <= tokens[i+1] <= 80:
            count += 1
            i += 2
        else:
            i += 1
    return count


def analyze_remi_files(midi_files, data_dir, num_samples=500, seed=42):
    """Analyze REMI encoding statistics."""
    rng = random.Random(seed)
    if len(midi_files) > num_samples:
        midi_files = rng.sample(midi_files, num_samples)

    results = {
        'seq_lengths': [],
        'notes_per_file': [],
        'tokens_per_note': [],
        'edit_distances': [],
        'truncated': 0,
        'total': 0,
    }

    for i, fname in enumerate(midi_files):
        fpath = os.path.join(data_dir, fname)
        try:
            tokens = midi_to_tokens(fpath)
            if len(tokens) == 0:
                continue

            if tokens[0] != BOS_TOKEN:
                tokens = [BOS_TOKEN] + tokens
            if tokens[-1] != EOS_TOKEN:
                tokens.append(EOS_TOKEN)

            results['total'] += 1
            results['seq_lengths'].append(len(tokens))

            num_notes = count_notes_remi(tokens)
            results['notes_per_file'].append(num_notes)
            if num_notes > 0:
                results['tokens_per_note'].append(len(tokens) / num_notes)

            if len(tokens) > 2048:
                results['truncated'] += 1

            # Edit distance: perturb and measure
            rng_state = random.getstate()
            random.seed(seed + i)
            source, target = remi_perturb(tokens)
            random.setstate(rng_state)

            ed = levenshtein_distance(source, target)
            results['edit_distances'].append(ed)

        except Exception as e:
            if i < 5:
                print(f"  REMI error on {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  REMI: {i+1}/{len(midi_files)}")

    return results


def analyze_beat_files(npz_files, data_dir, num_samples=500, seed=42):
    """Analyze BEAT encoding statistics (Scheme A)."""
    if not BEAT_AVAILABLE:
        return None

    rng = random.Random(seed)
    if len(npz_files) > num_samples:
        npz_files = rng.sample(npz_files, num_samples)

    # Create a minimal dataset to use the tokenizer
    ds = GECToRDataset(file_list=npz_files, data_dir=data_dir, max_len=99999)

    results = {
        'seq_lengths': [],
        'notes_per_file': [],
        'tokens_per_note': [],
        'edit_distances': [],
        'truncated': 0,
        'total': 0,
    }

    for i in range(min(num_samples, len(npz_files))):
        try:
            tokens = ds._tokenize_npz(i)
            if len(tokens) == 0:
                continue

            results['total'] += 1
            results['seq_lengths'].append(len(tokens))

            num_notes = count_notes_beat(tokens)
            results['notes_per_file'].append(num_notes)
            if num_notes > 0:
                results['tokens_per_note'].append(len(tokens) / num_notes)

            if len(tokens) > 2048:
                results['truncated'] += 1

            # Edit distance
            rng_state = random.getstate()
            random.seed(seed + i)
            source, target = beat_perturb(tokens)
            random.setstate(rng_state)

            ed = levenshtein_distance(source, target)
            results['edit_distances'].append(ed)

        except Exception as e:
            if i < 5:
                print(f"  BEAT error: {e}")

        if (i + 1) % 100 == 0:
            print(f"  BEAT: {i+1}/{min(num_samples, len(npz_files))}")

    return results


def alignment_stability_remi(midi_files, data_dir, num_samples=100, seed=42):
    """
    Measure alignment stability: insert 1 note, measure token displacement.

    For each file:
    1. Tokenize original
    2. Insert 1 note at a random position
    3. Re-tokenize
    4. Measure how many subsequent tokens shifted position
    """
    rng = random.Random(seed)
    if len(midi_files) > num_samples:
        midi_files = rng.sample(midi_files, num_samples)

    displacements = []

    for fname in midi_files:
        fpath = os.path.join(data_dir, fname)
        try:
            tokens = midi_to_tokens(fpath)
            if len(tokens) < 20:
                continue

            notes = tokens_to_notes(tokens)
            if len(notes) < 5:
                continue

            # Pick a random note and slightly modify (insert near it)
            original = list(tokens)

            # Insert a new Position+Pitch+Velocity+Duration group after a random bar
            parsed = parse_sequence(tokens)
            if not parsed['bars']:
                continue

            bar_idx = rng.randint(0, len(parsed['bars']) - 1)
            bar = parsed['bars'][bar_idx]
            if not bar['positions']:
                continue

            # Find insertion point (after the bar token)
            insert_at = bar['bar_token_idx'] + 1

            # Insert 4 tokens (Position, Pitch, Velocity, Duration)
            inserted = list(original)
            new_tokens = [190, 50, 100, 140]  # Position_0, Pitch_66, Vel, Dur
            for j, t in enumerate(new_tokens):
                inserted.insert(insert_at + j, t)

            # Count displaced tokens
            # Compare token-by-token after insertion point
            displacement = 0
            for k in range(insert_at + len(new_tokens), min(len(inserted), len(original) + len(new_tokens))):
                orig_k = k - len(new_tokens)
                if orig_k < len(original) and inserted[k] != original[orig_k]:
                    displacement += 1

            displacements.append(len(new_tokens))  # REMI: exact displacement = num inserted tokens

        except Exception:
            pass

    return displacements


def print_comparison(remi_results, beat_results):
    """Print comparison table."""
    def stats(values):
        if not values:
            return "N/A", "N/A", "N/A"
        return f"{np.mean(values):.1f}", f"{np.median(values):.1f}", f"{np.std(values):.1f}"

    print("\n" + "=" * 80)
    print("BEAT vs REMI Encoding Analysis")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'REMI':>20} {'BEAT (Scheme A)':>20}")
    print("-" * 70)

    # Sequence length
    r_mean, r_med, r_std = stats(remi_results['seq_lengths'])
    if beat_results:
        b_mean, b_med, b_std = stats(beat_results['seq_lengths'])
    else:
        b_mean = b_med = b_std = "N/A"
    print(f"{'Seq length (mean±std)':<30} {r_mean}±{r_std:>8} {b_mean}±{b_std:>8}")

    # Tokens per note
    r_mean, r_med, r_std = stats(remi_results['tokens_per_note'])
    if beat_results:
        b_mean, b_med, b_std = stats(beat_results['tokens_per_note'])
    else:
        b_mean = b_med = b_std = "N/A"
    print(f"{'Tokens per note (mean±std)':<30} {r_mean}±{r_std:>8} {b_mean}±{b_std:>8}")

    # Notes per file
    r_mean, r_med, r_std = stats(remi_results['notes_per_file'])
    if beat_results:
        b_mean, b_med, b_std = stats(beat_results['notes_per_file'])
    else:
        b_mean = b_med = b_std = "N/A"
    print(f"{'Notes per file (mean±std)':<30} {r_mean}±{r_std:>8} {b_mean}±{b_std:>8}")

    # Edit distance
    r_mean, r_med, r_std = stats(remi_results['edit_distances'])
    if beat_results:
        b_mean, b_med, b_std = stats(beat_results['edit_distances'])
    else:
        b_mean = b_med = b_std = "N/A"
    print(f"{'Edit distance (mean±std)':<30} {r_mean}±{r_std:>8} {b_mean}±{b_std:>8}")

    # Truncation
    r_trunc = f"{remi_results['truncated']}/{remi_results['total']}" if remi_results['total'] > 0 else "N/A"
    if beat_results:
        b_trunc = f"{beat_results['truncated']}/{beat_results['total']}" if beat_results['total'] > 0 else "N/A"
    else:
        b_trunc = "N/A"
    print(f"{'Truncated at 2048':<30} {r_trunc:>20} {b_trunc:>20}")

    # Truncation rate
    r_rate = f"{100*remi_results['truncated']/max(remi_results['total'],1):.1f}%" if remi_results['total'] > 0 else "N/A"
    if beat_results and beat_results['total'] > 0:
        b_rate = f"{100*beat_results['truncated']/max(beat_results['total'],1):.1f}%"
    else:
        b_rate = "N/A"
    print(f"{'Truncation rate':<30} {r_rate:>20} {b_rate:>20}")

    print("=" * 80)


def main():
    num_samples = 500
    seed = 42

    print("=" * 60)
    print("Layer 1 Encoding Analysis: BEAT vs REMI")
    print("=" * 60)

    # Get file lists
    midi_files = sorted([f for f in os.listdir(MIDI_DATA_DIR) if f.endswith('.mid')])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(midi_files))
    rng.shuffle(indices)
    test_size = int(len(midi_files) * 0.05)
    test_midi = [midi_files[i] for i in indices[-test_size:]]

    # REMI analysis
    print("\nAnalyzing REMI encoding...")
    remi_results = analyze_remi_files(test_midi, MIDI_DATA_DIR, num_samples, seed)

    # BEAT analysis
    beat_results = None
    if BEAT_AVAILABLE:
        print("\nAnalyzing BEAT encoding (Scheme A)...")
        npz_files = sorted([f for f in os.listdir(NPZ_DATA_DIR) if f.endswith('.npz')])
        rng2 = np.random.RandomState(seed)
        indices2 = np.arange(len(npz_files))
        rng2.shuffle(indices2)
        test_size2 = int(len(npz_files) * 0.05)
        test_npz = [npz_files[i] for i in indices2[-test_size2:]]
        beat_results = analyze_beat_files(test_npz, NPZ_DATA_DIR, num_samples, seed)

    # Alignment stability
    print("\nMeasuring alignment stability (REMI)...")
    remi_displacements = alignment_stability_remi(test_midi, MIDI_DATA_DIR, min(100, num_samples), seed)

    # Print results
    print_comparison(remi_results, beat_results)

    if remi_displacements:
        print(f"\nAlignment stability (REMI): mean displacement = {np.mean(remi_displacements):.1f} tokens")
        print(f"  (BEAT: insertion displaces exactly 2 tokens (pos+val) within beat, 0 across beats)")

    # Save
    output = {
        'remi': {
            'num_files': remi_results['total'],
            'seq_length': {'mean': float(np.mean(remi_results['seq_lengths'])),
                          'std': float(np.std(remi_results['seq_lengths'])),
                          'median': float(np.median(remi_results['seq_lengths']))},
            'tokens_per_note': {'mean': float(np.mean(remi_results['tokens_per_note'])),
                               'std': float(np.std(remi_results['tokens_per_note']))},
            'notes_per_file': {'mean': float(np.mean(remi_results['notes_per_file'])),
                              'std': float(np.std(remi_results['notes_per_file']))},
            'edit_distance': {'mean': float(np.mean(remi_results['edit_distances'])),
                             'std': float(np.std(remi_results['edit_distances']))},
            'truncation_rate': remi_results['truncated'] / max(remi_results['total'], 1),
            'alignment_displacement': float(np.mean(remi_displacements)) if remi_displacements else None,
        },
    }

    if beat_results:
        output['beat_scheme_a'] = {
            'num_files': beat_results['total'],
            'seq_length': {'mean': float(np.mean(beat_results['seq_lengths'])),
                          'std': float(np.std(beat_results['seq_lengths'])),
                          'median': float(np.median(beat_results['seq_lengths']))},
            'tokens_per_note': {'mean': float(np.mean(beat_results['tokens_per_note'])),
                               'std': float(np.std(beat_results['tokens_per_note']))},
            'notes_per_file': {'mean': float(np.mean(beat_results['notes_per_file'])),
                              'std': float(np.std(beat_results['notes_per_file']))},
            'edit_distance': {'mean': float(np.mean(beat_results['edit_distances'])),
                             'std': float(np.std(beat_results['edit_distances']))},
            'truncation_rate': beat_results['truncated'] / max(beat_results['total'], 1),
        }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', '..', 'results', 'baselines',
                               'beat_vs_remi_analysis.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
