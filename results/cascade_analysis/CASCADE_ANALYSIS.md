# Cascade Error Analysis: Complete Evidence Package

## Overview

This analysis provides systematic evidence for the **cascade error effect** in relative-position separated encoding (Scheme B). The cascade occurs because relative position tokens encode inter-note intervals as a cumulative sum — modifying one token shifts ALL downstream notes in the same beat.

## Files in this package

| File | Content |
|------|---------|
| `CASCADE_ANALYSIS.md` | This document |
| `injection_results.json` | Single-token injection experiment (Part 1) |
| `per_sample_analysis.json` | Per-sample evaluation statistics (Part 2) |
| `cross_scheme_failures.json` | Cross-scheme failure isolation (Part 3) |
| `recovery_distributions.json` | Raw recovery values for histogram (Part 4) |
| `recovery_histogram.pdf/png` | Recovery distribution figure |
| `cascade_by_position.pdf/png` | Cascade spread by injection position figure |

---

## Part 1: Token-Level Injection Experiment

**Method**: Generate 2000 random beats (2-10 notes each). For each non-last position token, apply SHIFT+1 and count how many notes change after decoding.

### Results (9887 / 1572 valid trials)

| Scheme | Mean Changed | Cascade Rate | Distribution |
|--------|:---:|:---:|:---|
| **B (Rel, Sep)** | **4.68** | **100.0%** | 1 note: 0%, 2-3: 39%, 4-5: 27%, 6+: 34% |
| **A (Abs, Sep)** | **1.00** | **0.0%** | 1 note: 100%, 2+: 0% |

**Key finding**: In Scheme B, a single SHIFT error *always* cascades (100% rate), affecting 4.68 notes on average. In Scheme A, every SHIFT error is perfectly isolated (exactly 1 note, 0% cascade). The cascade is a deterministic structural property of relative position encoding.

### Cascade by error position

Earlier errors in a beat cascade to more notes:

| Injection Position | B: Notes Affected | A: Notes Affected |
|:---:|:---:|:---:|
| Note 1 (first) | 6.00 | 1.00 |
| Note 2 | 5.50 | 1.00 |
| Note 3 | 5.00 | 1.00 |
| Note 4 | 4.54 | 1.00 |
| Note 5 | 4.03 | 1.00 |
| Note 6 | 3.56 | 1.00 |
| Note 7 | 3.01 | 1.00 |
| Note 8 | 2.52 | 1.00 |

The cascade spread equals the number of downstream notes — every downstream note shifts by the same delta. This is because `abs_pitch = cumsum(rel_deltas)`, so perturbing one delta offsets all subsequent absolute pitches.

### Concrete example

```
Beat: 5 notes at pitches [10, 20, 35, 55, 70]
Tokens (B): [91, 40, 91, 50, 96, 60, 101, 30, 96, 20, END]
           rel=10    rel=10   rel=15   rel=20   rel=15

Inject SHIFT+1 on Note 1 (token 91→92, rel 10→11):
Decoded:  [11, 21, 36, 56, 71]  ← ALL 5 notes shifted by +1
Changed: 5/5

Same beat in Scheme A:
Tokens (A): [91, 40, 101, 50, 116, 60, 136, 30, 151, 20]
           abs=10    abs=20   abs=35   abs=55   abs=70

Inject SHIFT+1 on Note 1 (token 91→92, abs 10→11):
Decoded:  [11, 20, 35, 55, 70]  ← Only note 1 changed
Changed: 1/5
```

---

## Part 2: End-to-End Cascade Evidence (200-sample evaluation)

### Error amplification

| Scheme | Orig Errors | After Model | Amplification | % Negative Recovery |
|:---:|:---:|:---:|:---:|:---:|
| A | 63.7 | 33.1 | 0.519 | 0.0% |
| D | 79.1 | 41.6 | 0.526 | 0.5% |
| C | 79.1 | 51.0 | 0.645 | 6.0% |
| REMI | 52.3 | 47.1 | 0.900 | 1.0% |
| **B** | **64.4** | **65.2** | **1.012** | **12.0%** |

B is the **only** scheme with amplification > 1.0 (model introduces more errors than it fixes).

### Recovery distribution (percentiles)

| Scheme | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | +0.139 | +0.246 | +0.356 | +0.507 | +0.625 | +0.696 | +0.736 |
| D | +0.150 | +0.206 | +0.324 | +0.474 | +0.616 | +0.698 | +0.720 |
| C | -0.006 | +0.033 | +0.196 | +0.376 | +0.552 | +0.678 | +0.724 |
| REMI | +0.013 | +0.026 | +0.048 | +0.081 | +0.147 | +0.200 | +0.250 |
| **B** | **-3.697** | **-0.036** | **+0.170** | **+0.377** | **+0.555** | **+0.663** | **+0.716** |

**Bimodal failure**: B's P50 (0.377) is close to A's P50 (0.507) — 88% of samples work normally. But P5 is -3.697 vs A's +0.139. A small fraction of catastrophic failures drags the mean negative.

### Catastrophic pitch error

| Scheme | MPE > 1.0 semitone | Max MPE |
|:---:|:---:|:---:|
| A | 0/199 (0.0%) | 0.21 |
| D | 0/200 (0.0%) | 0.23 |
| C | 0/200 (0.0%) | 0.33 |
| REMI | 0/200 (0.0%) | 0.35 |
| **B** | **11/200 (5.5%)** | **8.20** |

Only Scheme B produces catastrophic pitch errors. In the worst case, the average pitch error is 8.2 semitones — nearly an octave. The note count is preserved (ratio ≈ 0.95), confirming the model produces the right number of notes at completely wrong positions.

---

## Part 3: Cross-Scheme Failure Isolation

18 samples fail catastrophically under B (recovery < -10%) but succeed under A/C/D:

| Metric | B | A | C | D |
|--------|:---:|:---:|:---:|:---:|
| Mean recovery | **-4.146** | +0.492 | +0.384 | +0.448 |

Same files, same perturbation, different encoding → the failure is **encoding-specific**.

Top 5 worst B failures (all recover normally under A):

| File | B recovery | A recovery | D recovery |
|------|:---:|:---:|:---:|
| 117970.npz | -8.719 | +0.433 | +0.415 |
| 1260556.npz | -8.418 | +0.078 | +0.083 |
| 1147081.npz | -7.897 | +0.335 | +0.325 |
| 1241516.npz | -7.079 | +0.458 | +0.412 |
| 1028626.npz | -5.809 | +0.749 | +0.778 |

---

## Part 4: Complete Evidence Chain

```
Theory (§2.2)
  Relative position = cumulative sum → single error shifts all downstream notes
    ↓
Token Injection (this analysis, Part 1)
  B: 1 error → 4.68 notes (100% cascade rate)
  A: 1 error → 1.00 note  (0% cascade rate)
    ↓
SHIFT Design (§3.1)
  Added ±1~5 SHIFT labels to compensate, but model must predict TWO
  simultaneous SHIFTs (edit + compensation) — any miss cascades
    ↓
Training Paradox
  B: MLM PPL=1.25 (best) → EditF1=0.863 (best) → recovery=-0.8% (worst)
  Each component is individually optimal, but cascade at deployment kills performance
    ↓
End-to-End (Part 2)
  B: 12% samples worsened, 5.5% catastrophic (MPE > 1 semitone)
  Error amplification = 1.012 (only scheme > 1.0)
    ↓
Cross-File Isolation (Part 3)
  18 catastrophic B failures achieve normal recovery under A/C/D
  → Failure is purely encoding-specific, not data-dependent
```

## Bundled Encoding: Structural Solution

Schemes C/D (bundled, 1 token/note) have **no SHIFT operation at all** — they use REPLACE. One REPLACE label changes one entire note (position + value) with zero side effects. This is why:
- D (Abs, Bun): 0.5% negative, 0% catastrophic
- C (Rel, Bun): 6.0% negative, 0% catastrophic — mild cascade from relative position within bundled token, but no inter-token propagation

The 2×2 encoding design space thus reveals a clear hierarchy for error isolation:
```
D (abs + bundled): zero cascade, zero catastrophic
A (abs + separated): zero cascade, zero catastrophic
C (rel + bundled): mild (6% negative, 0% catastrophic)
B (rel + separated): severe (12% negative, 5.5% catastrophic)
```

## Paper Usage Suggestions

### For Discussion section (~half page):
- Title: "The cascade paradox: when better components produce worse systems"
- Use the injection experiment numbers (B: 4.68x, A: 1.00x) as the core evidence
- Cross-scheme table (18 files) as definitive proof of encoding-specificity
- Brief mention of bimodal distribution and catastrophic failure rate

### For Appendix (~1 page):
- Full injection experiment table with by-position breakdown
- Recovery histogram figure (recovery_histogram.pdf)
- Cascade-by-position figure (cascade_by_position.pdf)
- Complete cross-scheme failure table (18 files × 4 schemes)
- Percentile table showing bimodal distribution

### Key LaTeX-ready numbers:
- `1 \text{ error} \to 4.68 \text{ notes}` (B cascade amplification)
- `100\%` cascade rate for B vs `0\%` for A
- `12\%` negative recovery for B vs `0\%$ for A
- `5.5\%` catastrophic pitch errors (MPE > 1 semitone) for B, `0\%` for all others
- 18 cross-scheme failures: B recovery $-4.15$ vs A recovery $+0.49$
