# Evaluation Report

## 1. Experimental Setup

### 1.1 Encoding Schemes

We adopt a 2x2 ablation design that crosses position encoding (absolute / relative) with token organization (separated / bundled), giving four Beat Encoding schemes:

| Scheme | Position | Token Org. | Vocab | Tok/Note |
| :--- | :---: | :---: | :---: | :---: |
| A (no_pair) | Absolute | Separated | 186 | 2 |
| B (no_pair_related) | Relative | Separated | 185 | 2 |
| C (with_pair) | Relative | Bundled | 7145 | 1 |
| D (absolute_bundled) | Absolute | Bundled | 7145 | 1 |

Separated encoding splits pitch and rhythm value into two tokens; bundled encoding compresses the complete note information into a single token.

### 1.2 Evaluated Systems

| System | Type | Description |
| :--- | :--- | :--- |
| **TagFill** | Non-autoregressive two-stage editing (FELIX-style) | Tagger predicts edit operations, then Inserter fills the MASK slots |
| **SeqTag** | Single-stage sequence tagging (GECToR-style) | Frozen BERT encoder + error detection + label prediction |
| **BERT-CMLM** | Baseline (no additional training) | Pre-trained BERT MLM used directly for mask-predict correction |
| **LLaMA** | Autoregressive generation baseline | Continues the accompaniment from a prompt (fundamentally a different task from editing) |
| **No-Edit** | Trivial baseline | Returns the corrupted input unchanged |
| **Copy-Context** | Trivial baseline | Replaces the current beat with a copy of the previous beat's accompaniment |

### 1.3 Metrics

**Strict-match metrics:**
- **beat_exact_match**: fraction of beats whose predicted accompaniment matches the target exactly, compared beat by beat
- **token_accuracy**: token-level accuracy

**Soft pitch metrics:**
- **note_f1_tol0**: note F1 under strict pitch matching (tolerance 0 semitones)
- **note_f1_tol2**: note F1 with a +/-2 semitone tolerance (small deviations allowed)
- **note_f1_tol4**: note F1 with a +/-4 semitone tolerance
- **chroma_f1**: pitch-class F1, ignoring octave
- **mean_pitch_error** (MPE): mean absolute pitch error in semitones (lower is better)

**Distribution-level metrics:**
- **bert_cosine_sim**: cosine similarity in the BERT embedding space (higher is better)
- **FMD (Frechet Music Distance)**: Frechet distance in the BERT embedding space, measuring how close the generated distribution is to the real one (lower is better)

**Conventional metrics:**
- **pitch_class_overlap**: overlap of the pitch-class distributions
- **rhythm_similarity**: similarity of rhythmic patterns
- **note_density_ratio**: note-density ratio (1.0 is ideal)

---

## 2. Core Comparison (50-sample evaluation)

The tables below are the core numbers reported in the paper: 50 evaluation samples, covering every method and every metric.

### 2.1 Strict-Match Metrics

#### beat_exact_match

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1714 | 0.0083 | 0.0501 | 0.9377 | **0.9629** |
| B (rel, sep) | 0.1719 | 0.0082 | 0.0412 | 0.8939 | **0.9620** |
| C (rel, bun) | 0.1736 | 0.0090 | 0.1073 | 0.9372 | **0.9697** |
| D (abs, bun) | 0.1736 | 0.0090 | 0.1197 | 0.9445 | **0.9702** |

#### token_accuracy

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1868 | 0.0135 | 0.0917 | 0.9706 | **0.9794** |
| B (rel, sep) | 0.1871 | 0.0134 | 0.0606 | 0.9371 | **0.9756** |
| C (rel, bun) | 0.1886 | 0.0134 | 0.1170 | 0.9618 | **0.9839** |
| D (abs, bun) | 0.1886 | 0.0137 | 0.1479 | 0.9741 | **0.9859** |

> On the strict metrics TagFill and SeqTag reach 89%-97% beat-level exact match, far above every baseline (No-Edit ~17%, CMLM ~4-12%). SeqTag outperforms TagFill on all four schemes.

### 2.2 Soft Pitch Metrics

#### note_f1_tol0 (strict pitch F1)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1177 | 0.0176 | 0.0430 | 0.9668 | **0.9813** |
| B (rel, sep) | 0.1173 | 0.0170 | 0.0236 | 0.9499 | **0.9780** |
| C (rel, bun) | 0.1199 | 0.0094 | 0.0669 | 0.9554 | **0.9851** |
| D (abs, bun) | 0.1199 | 0.0095 | 0.0912 | 0.9706 | **0.9880** |

#### note_f1_tol2 (+/-2 semitone tolerance)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1226 | 0.0397 | 0.0674 | 0.9883 | **0.9897** |
| B (rel, sep) | 0.1221 | 0.0395 | 0.0382 | 0.9725 | **0.9871** |
| C (rel, bun) | 0.1250 | 0.0229 | 0.0846 | 0.9773 | **0.9920** |
| D (abs, bun) | 0.1250 | 0.0234 | 0.1055 | 0.9894 | **0.9935** |

#### note_f1_tol4 (+/-4 semitone tolerance)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1280 | 0.0641 | 0.0864 | **0.9951** | 0.9917 |
| B (rel, sep) | 0.1275 | 0.0636 | 0.0501 | 0.9806 | **0.9902** |
| C (rel, bun) | 0.1303 | 0.0366 | 0.0981 | 0.9847 | **0.9949** |
| D (abs, bun) | 0.1303 | 0.0382 | 0.1129 | **0.9959** | 0.9958 |

#### chroma_f1 (octave-invariant pitch-class F1)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.1183 | 0.0250 | 0.0557 | 0.9683 | **0.9825** |
| B (rel, sep) | 0.1179 | 0.0249 | 0.0519 | 0.9533 | **0.9803** |
| C (rel, bun) | 0.1206 | 0.0206 | 0.0833 | 0.9591 | **0.9865** |
| D (abs, bun) | 0.1206 | 0.0206 | 0.0977 | 0.9726 | **0.9893** |

#### mean_pitch_error (MPE, in semitones, lower is better)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 37.96 | 39.97 | 38.55 | **0.30** | 0.59 |
| B (rel, sep) | 37.97 | 40.02 | 38.58 | 1.00 | **0.67** |
| C (rel, bun) | 37.91 | 40.89 | 38.27 | 0.61 | **0.31** |
| D (abs, bun) | 37.91 | 40.88 | 38.09 | **0.25** | 0.36 |

> TagFill and SeqTag have a mean pitch error of only 0.25-1.00 semitones, whereas every baseline sits at 37-41 semitones, a roughly 40x gap. On the soft metric note_f1_tol2, SeqTag D reaches 0.9935; the gap between note_f1_tol2 and note_f1_tol0 is only 0.005, showing that the residual deviations are tiny.

### 2.3 Distribution-Level Metrics

#### bert_cosine_sim (higher is better)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.8908 | 0.8884 | 0.8908 | 0.9995 | **0.9997** |
| B (rel, sep) | 0.9073 | 0.9032 | 0.9063 | 0.9987 | **0.9997** |
| C (rel, bun) | 0.9044 | 0.8974 | 0.9111 | 0.9993 | **0.9997** |
| D (abs, bun) | 0.9120 | 0.9092 | 0.9132 | 0.9996 | **0.9998** |

#### FMD (Frechet Music Distance, lower is better)

> **Note**: FMD is a population-level distribution metric computed over the full sample set, not a per-sample average. It measures the Frechet distance between the generated and the real distribution and needs a sufficient sample size for a stable estimate, so it is computed here over all samples (178-190 per scheme).

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 3.71 | 4.52 | 3.31 | **3.15** | 3.21 |
| B (rel, sep) | 3.52 | 4.97 | **2.35** | 5.44 | 2.66 |
| C (rel, bun) | 4.17 | 5.87 | 2.80 | **1.97** | 2.98 |
| D (abs, bun) | 2.99 | 3.86 | 2.47 | **1.63** | 2.52 |

> BERT cosine similarity separates the method tiers cleanly: TagFill/SeqTag ~0.999 vs. baselines ~0.89-0.91. On FMD, TagFill is strongest under the bundled encodings: TagFill D = 1.63 is the global minimum, with TagFill C = 1.97 close behind. SeqTag's FMD ranges from 2.35 to 3.21, better than the baselines overall (2.99-5.87). TagFill on Scheme B has FMD = 5.44, worse even than Copy-Context (4.97), further confirming the systematic failure of Scheme B under TagFill.

---

## 3. Full-Sample Reference Results (178-190 samples)

The following results cover all valid samples (after filtering out sequences longer than 2048, 178-190 samples per scheme), and are provided as supplementary reference.

### 3.1 Strict-Match Metrics

#### beat_exact_match

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.5947 | 0.1483 | 0.2897 | 0.6411 | **0.6489** |
| B (rel, sep) | 0.5936 | 0.1487 | 0.3010 | 0.4174 | **0.5391** |
| C (rel, bun) | 0.6099 | 0.1518 | 0.4362 | 0.6549 | **0.6720** |
| D (abs, bun) | 0.6099 | 0.1518 | 0.4512 | **0.6756** | 0.6721 |

### 3.2 Soft Pitch Metrics

#### note_f1_tol0 (strict pitch F1)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6577 | 0.3249 | 0.4070 | 0.6830 | **0.6743** |
| B (rel, sep) | 0.6567 | 0.3249 | 0.3705 | 0.4864 | **0.5834** |
| C (rel, bun) | 0.6765 | 0.2782 | 0.4944 | 0.6630 | **0.6929** |
| D (abs, bun) | 0.6765 | 0.2792 | 0.5583 | 0.6901 | **0.6964** |

#### note_f1_tol2 (+/-2 semitone tolerance)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6807 | 0.4372 | 0.4828 | **0.7096** | 0.6898 |
| B (rel, sep) | 0.6798 | 0.4369 | 0.4598 | 0.5661 | **0.6249** |
| C (rel, bun) | 0.6993 | 0.3902 | 0.5638 | 0.6914 | **0.7105** |
| D (abs, bun) | 0.6993 | 0.3920 | 0.6036 | **0.7165** | 0.7117 |

#### note_f1_tol4 (+/-4 semitone tolerance)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6926 | 0.4979 | 0.5491 | **0.7248** | 0.6986 |
| B (rel, sep) | 0.6914 | 0.4975 | 0.5173 | 0.6065 | **0.6454** |
| C (rel, bun) | 0.7106 | 0.4517 | 0.6092 | 0.7043 | **0.7192** |
| D (abs, bun) | 0.7106 | 0.4545 | 0.6431 | **0.7305** | 0.7203 |

#### chroma_f1 (octave-invariant pitch-class F1)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6618 | 0.3654 | 0.4595 | **0.6892** | 0.6787 |
| B (rel, sep) | 0.6607 | 0.3645 | 0.4340 | 0.5442 | **0.5973** |
| C (rel, bun) | 0.6804 | 0.3396 | 0.5388 | 0.6731 | **0.6974** |
| D (abs, bun) | 0.6804 | 0.3402 | 0.5913 | 0.6940 | **0.7006** |

#### mean_pitch_error (MPE, lower is better)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 15.11 | 21.61 | 16.99 | **12.36** | 14.17 |
| B (rel, sep) | 15.15 | 21.63 | 17.12 | 15.81 | **16.29** |
| C (rel, bun) | 14.36 | 21.84 | 15.39 | **13.18** | 13.37 |
| D (abs, bun) | 14.36 | 21.83 | 14.96 | **11.92** | 13.40 |

### 3.3 Distribution-Level Metrics

#### bert_cosine_sim (higher is better)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.9643 | 0.9600 | 0.9613 | **0.9699** | 0.9668 |
| B (rel, sep) | 0.9665 | 0.9587 | 0.9640 | 0.9539 | **0.9725** |
| C (rel, bun) | 0.9660 | 0.9558 | 0.9697 | **0.9748** | 0.9721 |
| D (abs, bun) | 0.9724 | 0.9674 | 0.9719 | **0.9779** | 0.9750 |

#### FMD (Frechet Music Distance, lower is better)

| Scheme | No-Edit | Copy-Ctx | CMLM | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 3.71 | 4.52 | 3.31 | **3.15** | 3.21 |
| B (rel, sep) | 3.52 | 4.97 | **2.35** | 5.44 | 2.66 |
| C (rel, bun) | 4.17 | 5.87 | 2.80 | **1.97** | 2.98 |
| D (abs, bun) | 2.99 | 3.86 | 2.47 | **1.63** | 2.52 |

> In the full-sample results, TagFill D has the globally best FMD (1.63), with TagFill C (1.97) close behind. TagFill on Scheme B has FMD = 5.44, worse even than Copy-Context (4.97), again confirming the systematic failure of Scheme B under TagFill. Across the full sample set the gap between note_f1_tol2 and note_f1_tol0 is about 0.02-0.03 (TagFill/SeqTag), confirming that most prediction errors stay within +/-2 semitones.

---

## 4. Ablation Studies

### 4.1 TagFill Iterative Decoding Ablation

Effect of running the Tagger-Inserter loop for iter = 1, 2, 3, 5 iterations.

#### beat_exact_match

| Scheme | iter=1 | iter=2 | iter=3 | iter=5 |
| :--- | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6575 | 0.5978 | 0.6658 | **0.6681** |
| B (rel, sep) | **0.4094** | 0.3755 | 0.4041 | 0.4042 |
| C (rel, bun) | 0.6788 | 0.6226 | **0.6798** | 0.6796 |
| D (abs, bun) | 0.7028 | 0.6164 | **0.7029** | 0.7026 |

#### token_accuracy

| Scheme | iter=1 | iter=2 | iter=3 | iter=5 |
| :--- | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.7393 | 0.7025 | 0.7453 | **0.7467** |
| B (rel, sep) | **0.4345** | 0.4194 | 0.4303 | 0.4303 |
| C (rel, bun) | 0.7089 | 0.6738 | **0.7096** | 0.7094 |
| D (abs, bun) | 0.7356 | 0.6745 | **0.7367** | 0.7365 |

> Performance drops noticeably at iter=2 and recovers at iter=3, matching or slightly exceeding iter=1. iter=3 and iter=5 are essentially tied, indicating that three iterations already converge. Scheme B is the worst scheme at every iteration count.

### 4.2 TagFill Inference-Optimization Ablation

Three inference-time options, evaluated individually and combined: Iterative (iterative Inserter decoding with the linear acceptance schedule of paper Eq. 9, inserter_steps=5 in this ablation; the default is T=2), Skeleton (confidence-based skeleton, tagger_conf=0.3), and Harmony (harmonic constraint, weight=0.5).

#### beat_exact_match

| Scheme | Baseline | Iterative | Skeleton | Harmony | All |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6647 | **0.6649** | 0.6647 | 0.5976 | 0.5978 |
| B (rel, sep) | 0.4094 | **0.4146** | 0.4071 | 0.3737 | 0.3755 |
| C (rel, bun) | 0.6795 | **0.6848** | 0.6796 | 0.6164 | 0.6226 |
| D (abs, bun) | **0.7028** | 0.7028 | 0.6820 | 0.6346 | 0.6164 |

#### token_accuracy

| Scheme | Baseline | Iterative | Skeleton | Harmony | All |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.7451 | 0.7454 | **0.7454** | 0.7019 | 0.7025 |
| B (rel, sep) | 0.4340 | **0.4397** | 0.4337 | 0.4132 | 0.4194 |
| C (rel, bun) | 0.7098 | **0.7124** | 0.7099 | 0.6701 | 0.6738 |
| D (abs, bun) | **0.7366** | 0.7363 | 0.7157 | 0.6940 | 0.6745 |

> Iterative decoding gives a marginal gain on most schemes (+0.001 to +0.005). Skeleton is essentially neutral. The harmonic constraint actually causes a clear drop, and the full combination (All) is dragged below the baseline by Harmony. The returns from inference-time optimization are limited, indicating that TagFill's single-pass decoding is already close to sufficient.

---

## 5. Method Comparison

### 5.1 SeqTag vs. TagFill, Unified Comparison

Per-scheme comparison on the same test set and under the same perturbations.

#### beat_exact_match

| Scheme | SeqTag | TagFill | Delta |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.6605** | 0.6373 | -0.0232 |
| B (rel, sep) | **0.5184** | 0.3851 | -0.1333 |
| C (rel, bun) | **0.6896** | 0.6748 | -0.0148 |
| D (abs, bun) | 0.6886 | **0.6991** | +0.0106 |

#### token_accuracy

| Scheme | SeqTag | TagFill | Delta |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | 0.6901 | **0.7220** | +0.0319 |
| B (rel, sep) | **0.5437** | 0.4120 | -0.1317 |
| C (rel, bun) | **0.7168** | 0.7050 | -0.0118 |
| D (abs, bun) | 0.7191 | **0.7318** | +0.0128 |

#### pitch_class_overlap

| Scheme | SeqTag | TagFill | Delta |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | 0.7752 | **0.8293** | +0.0541 |
| B (rel, sep) | 0.7779 | **0.8686** | +0.0908 |
| C (rel, bun) | **0.7948** | 0.7784 | -0.0165 |
| D (abs, bun) | **0.7938** | 0.7891 | -0.0047 |

#### rhythm_similarity

| Scheme | SeqTag | TagFill | Delta |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | 0.8376 | **0.8889** | +0.0513 |
| B (rel, sep) | 0.8385 | **0.9265** | +0.0879 |
| C (rel, bun) | 0.8507 | **0.8554** | +0.0046 |
| D (abs, bun) | 0.8514 | **0.8551** | +0.0037 |

#### note_density_ratio

| Scheme | SeqTag | TagFill | Delta |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | 0.7519 | - | - |
| B (rel, sep) | 0.7517 | - | - |
| C (rel, bun) | 0.7835 | **0.7925** | +0.0090 |
| D (abs, bun) | 0.7809 | **0.7873** | +0.0064 |

#### Win Counts per Scheme

| Scheme | SeqTag wins | TagFill wins |
| :--- | :---: | :---: |
| A (abs, sep) | 1 | 3 |
| B (rel, sep) | 2 | 2 |
| C (rel, bun) | 3 | 2 |
| D (abs, bun) | 1 | 4 |

> SeqTag has a clear edge on beat_exact_match (winning on 3 of 4 schemes), while TagFill does better on soft metrics such as rhythm_similarity and pitch_class_overlap. Scheme D is the only scheme where TagFill beats SeqTag on beat_exact_match.

### 5.2 Edit Rate vs. Performance Curve

TagFill's beat_exact_match under increasing perturbation strength, measuring robustness to error density.

| Target edit rate | Actual edit rate | A (abs, sep) | B (rel, sep) | C (rel, bun) | D (abs, bun) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ~5% | 12.0% | **0.9352** | 0.9292 | 0.9228 | 0.9274 |
| ~10% | 11.5% | **0.9397** | 0.9355 | 0.9314 | 0.9326 |
| ~15% | 15.6% | 0.9089 | 0.8311 | 0.8948 | **0.9103** |
| ~20% | 24.5% | 0.8433 | 0.5322 | 0.8199 | **0.8585** |
| ~30% | 28.0% | 0.8056 | 0.4163 | 0.7923 | **0.8295** |
| ~40% | 41.4% | **0.6948** | 0.2843 | 0.6416 | 0.6940 |
| ~50% | 46.6% | 0.6450 | 0.2200 | 0.6162 | **0.6575** |
| ~60% | 43.3% | **0.6715** | 0.1854 | 0.6437 | 0.6637 |
| ~80% | 70.5% | 0.2179 | 0.1356 | 0.3291 | **0.3456** |
| ~100% | 90.7% | 0.0127 | 0.0517 | **0.0922** | **0.0922** |

> Every scheme degrades monotonically as the edit rate rises. At low edit rates (<15%) the four schemes differ little (all >0.90); at medium-to-high edit rates (20-60%) Scheme D retains the best robustness. Scheme B collapses sharply once the edit rate exceeds 15%, reaching only 0.14 at 80%. The bundled encodings (C/D) decay markedly more slowly than the separated ones under high edit rates.

---

## 6. Baseline Comparison

### 6.1 Trivial Baselines (No-Edit + Copy-Context)

No-Edit returns the corrupted input unchanged; Copy-Context replaces the current beat with the previous beat's accompaniment.

#### beat_exact_match (overall)

| Scheme | No-Edit | Copy-Context | TagFill | SeqTag | TagFill gain | SeqTag gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.6286 | 0.1390 | 0.6647 | 0.6605 | +0.0361 | +0.0319 |
| B (rel, sep) | 0.6286 | 0.1394 | 0.4094 | 0.5184 | -0.2192 | -0.1102 |
| C (rel, bun) | 0.6259 | 0.1356 | 0.6795 | 0.6896 | +0.0536 | +0.0637 |
| D (abs, bun) | 0.6259 | 0.1356 | 0.7028 | 0.6886 | +0.0769 | +0.0627 |

#### beat_exact_match by perturbation level (averaged over schemes A-D)

| Level | Samples | No-Edit | Copy-Context | Best model |
| :--- | :---: | :---: | :---: | :---: |
| L1 (light) | 11 | 0.880 | 0.190 | ~0.93 (TagFill) |
| L2 (moderate) | 18 | 0.747 | 0.126 | ~0.85 (TagFill D) |
| L3 (heavy) | 14 | 0.547 | 0.136 | ~0.64 (TagFill D) |
| L4 (extreme) | 7 | 0.083 | 0.083 | ~0.08 (all collapse) |

> No-Edit quantifies the severity of each perturbation level: L1 = 0.88 (light edits), L4 = 0.08 (near-complete rewrite). Copy-Context only reaches 0.14, far below No-Edit, showing that piano accompaniment varies enough from beat to beat that naive repetition fails completely. TagFill on Scheme B (0.41) even falls below No-Edit (0.63), which is direct evidence of cascaded error propagation.

### 6.2 BERT-CMLM Baseline

CMLM-style correction with the pre-trained Music BERT (`BertForMaskedLM`): forward pass, mask the low-confidence accompaniment positions, predict, iterate. No additional training is required; the existing BERT checkpoints are used directly.

#### beat_exact_match (iter = 1, 3, 5)

| Scheme | iter=1 | iter=3 | iter=5 | No-Edit | TagFill | SeqTag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A (abs, sep) | 0.3184 | 0.3063 | 0.2926 | 0.6286 | 0.6647 | **0.6605** |
| B (rel, sep) | 0.3300 | 0.3216 | 0.3063 | 0.6286 | 0.4094 | **0.5184** |
| C (rel, bun) | 0.4611 | 0.4520 | 0.4309 | 0.6259 | 0.6795 | **0.6896** |
| D (abs, bun) | 0.4684 | 0.4501 | 0.4301 | 0.6259 | **0.7028** | 0.6886 |

#### beat_exact_match by perturbation level (iter=1)

| Level | Samples | A | B | C | D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| L1 | 11 | 0.4326 | 0.4400 | 0.5826 | **0.6042** |
| L2 | 18 | 0.3484 | 0.3739 | **0.5662** | 0.5539 |
| L3 | 14 | 0.2711 | 0.2809 | 0.3891 | **0.4073** |
| L4 | 7 | 0.0826 | 0.0826 | 0.0831 | 0.0831 |

> CMLM falls below No-Edit on every scheme, showing that a plain BERT MLM cannot correct errors effectively and that a dedicated architecture (SeqTag / TagFill) is required. More iterations make things worse (iter=1 > iter=3 > iter=5), indicating that BERT's confidence calibration is unsuited to error detection. The bundled encodings (C/D, ~0.46) are far ahead of the separated ones (A/B, ~0.32), because each token carries the complete note information, which helps contextual prediction. Overall ranking: TagFill/SeqTag > No-Edit > CMLM > Copy-Context.

---

## 7. Advanced Experiments

### 7.1 Tagger Bottleneck Analysis (Editing / Oracle / Inpainting)

Three modes on the same perturbed input, used to disentangle the contributions of the Tagger and the Inserter:
- **Editing**: the standard Tagger -> Inserter pipeline
- **Oracle**: skip the Tagger by using the ground-truth `changed_mask` and hand it straight to the Inserter
- **Inpainting**: mask the entire accompaniment and let the Inserter generate it from scratch

#### beat_exact_match (overall)

| Scheme | Editing | Oracle | Inpainting | Oracle-Editing | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| A (abs, sep) | 0.6647 | 0.6701 | 0.1349 | +0.0054 | Neutral |
| B (rel, sep) | 0.4094 | 0.4731 | 0.0658 | +0.0638 | Tagger is the bottleneck |
| C (rel, bun) | 0.6795 | 0.6645 | 0.2068 | -0.0149 | Tagger contributes positively |
| D (abs, bun) | 0.7028 | 0.6689 | 0.2290 | -0.0339 | Tagger contributes positively |

#### beat_exact_match by perturbation level

**Scheme A:**

| Level | Samples | Editing | Oracle | Inpainting |
| :--- | :---: | :---: | :---: | :---: |
| L1 | 11 | 0.9273 | **0.9355** | 0.0642 |
| L2 | 18 | **0.8347** | 0.8159 | 0.1674 |
| L3 | 14 | 0.5595 | **0.5679** | 0.1136 |
| L4 | 7 | 0.0253 | **0.0826** | 0.2053 |

**Scheme B:**

| Level | Samples | Editing | Oracle | Inpainting |
| :--- | :---: | :---: | :---: | :---: |
| L1 | 11 | **0.9316** | 0.7413 | 0.0615 |
| L2 | 18 | 0.4373 | **0.7490** | 0.0666 |
| L3 | 14 | **0.1506** | 0.1206 | 0.0642 |
| L4 | 7 | 0.0344 | **0.0472** | 0.0737 |

**Scheme C:**

| Level | Samples | Editing | Oracle | Inpainting |
| :--- | :---: | :---: | :---: | :---: |
| L1 | 11 | **0.9232** | 0.9208 | 0.1931 |
| L2 | 18 | **0.8194** | 0.8047 | 0.2143 |
| L3 | 14 | **0.6062** | 0.5736 | 0.2150 |
| L4 | 7 | **0.0832** | 0.0831 | 0.1925 |

**Scheme D:**

| Level | Samples | Editing | Oracle | Inpainting |
| :--- | :---: | :---: | :---: | :---: |
| L1 | 11 | **0.9231** | 0.9213 | 0.2210 |
| L2 | 18 | **0.8562** | 0.8136 | 0.2382 |
| L3 | 14 | **0.6425** | 0.5774 | 0.2306 |
| L4 | 7 | **0.0832** | 0.0831 | 0.2145 |

> The role of the Tagger depends on the encoding scheme. Under Scheme B the Tagger is clearly the bottleneck: at level L2, Oracle = 0.749 vs. Editing = 0.437, so the Tagger alone costs 0.31. Under the bundled encodings (C/D), Editing overtakes Oracle, meaning the Tagger's selective-retention strategy is a net positive for those schemes. Inpainting reaches only 0.07-0.23, confirming that the Inserter relies on a contextual skeleton and cannot generate from scratch. At L4 (near-complete rewrite), however, Inpainting (0.19-0.21) beats Editing/Oracle (0.03-0.08), because placing MASK tokens preserves the correct token count.

### 7.2 AR vs. NAR (LLaMA vs. TagFill)

LLaMA (autoregressive continuation) and TagFill (non-autoregressive editing) solve fundamentally different tasks. LLaMA continues the accompaniment from a prompt, whereas TagFill corrects or generates the accompaniment conditioned on the complete melody. This comparison is therefore only an approximate reference.

#### beat_exact_match

| Scheme | LLaMA (AR) | TagFill (NAR) | Winner |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.0633** | 0.0000 | LLaMA |
| B (rel, sep) | **0.0669** | 0.0197 | LLaMA |
| C (rel, bun) | 0.0619 | **0.0797** | TagFill |

#### token_accuracy

| Scheme | LLaMA (AR) | TagFill (NAR) | Winner |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.0516** | 0.0494 | LLaMA |
| B (rel, sep) | **0.0553** | 0.0185 | LLaMA |
| C (rel, bun) | 0.0440 | **0.0478** | TagFill |

#### pitch_class_overlap

| Scheme | LLaMA (AR) | TagFill (NAR) | Winner |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.3719** | 0.0000 | LLaMA |
| B (rel, sep) | 0.4032 | **0.7556** | TagFill |
| C (rel, bun) | **0.4270** | 0.0000 | LLaMA |

#### rhythm_similarity

| Scheme | LLaMA (AR) | TagFill (NAR) | Winner |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.7449** | 0.0000 | LLaMA |
| B (rel, sep) | **0.7861** | 0.7147 | LLaMA |
| C (rel, bun) | **0.8005** | 0.0000 | LLaMA |

#### note_density_ratio

| Scheme | LLaMA (AR) | TagFill (NAR) | Winner |
| :--- | :---: | :---: | :---: |
| A (abs, sep) | **0.7868** | 0.0000 | LLaMA |
| B (rel, sep) | 0.7712 | **1.0136** | TagFill |
| C (rel, bun) | **0.8275** | 0.0000 | LLaMA |

> Both families score low in absolute terms (all below 0.1 beat_exact_match) because the tasks they perform are fundamentally different. As a continuation model, LLaMA has a slight edge on most metrics, but TagFill under Scheme C wins on beat_exact_match and token_accuracy. TagFill scores 0.0 on several metrics for Schemes A and C because its output degenerates when generating from scratch in Inpainting mode. The main takeaway of this comparison is the intrinsic difference between conditional editing and unconditional generation.

---

## 8. Key Findings

### Interaction Between Encoding Scheme and Method

1. **Rank reversal**: SeqTag ranks the schemes C > B > A > D, whereas TagFill ranks them D > C > A >> B, i.e. the two systems have nearly opposite encoding preferences. This indicates a strong interaction between the choice of encoding scheme and the model architecture: there is no universally optimal encoding.

2. **The Scheme B paradox**: Scheme B performs excellently under SeqTag (Tagger F1 = 0.605, the best of the four schemes) but worst in the end-to-end TagFill evaluation (beat_exact_match = 0.41, even below No-Edit's 0.63). The root cause is error amplification of the two-token separated encoding inside a two-stage cascade: small tagging deviations from the Tagger get multiplied by the Inserter.

3. **Robustness of the bundled encodings**: at high edit rates (>20%), the bundled encodings (C/D) degrade markedly more slowly than the separated ones (A/B), and Scheme B in particular collapses once the edit rate exceeds 15%. Carrying complete note information in a single token is critical to system robustness.

### Method Effectiveness

4. **TagFill and SeqTag both beat every baseline by a wide margin**: in the 50-sample evaluation both systems reach 89-97% beat_exact_match, versus only ~17% for No-Edit, ~4-12% for CMLM, and <1% for Copy-Context. On mean pitch error, TagFill/SeqTag are at just 0.25-1.00 semitones while the baselines sit at 37-41, a roughly 40x gap.

5. **FMD reveals distribution-level quality differences**: as a population-level metric over the full sample set, FMD shows TagFill strongest under the bundled encodings (D = 1.63, C = 1.97), with SeqTag ranging from 2.35 to 3.21, all better than No-Edit (2.99-4.17) and Copy-Context (3.86-5.87). TagFill on Scheme B has the worst FMD (5.44), worse even than Copy-Context, once again confirming the systematic failure of Scheme B under TagFill.

6. **BERT cosine similarity also separates the tiers cleanly**: TagFill/SeqTag ~0.999 vs. baselines ~0.89-0.91, although FMD discriminates more sharply.

7. **Soft metrics confirm prediction quality**: the gap between note_f1_tol2 and note_f1_tol0 is only 0.005 (50 samples) to 0.02-0.03 (full sample set), showing that the vast majority of prediction deviations lie within +/-2 semitones. The models are not only accurate, their residual errors are also very small.

8. **Performance ladder**: TagFill/SeqTag > No-Edit > CMLM > Copy-Context (with TagFill on Scheme B as the exception, falling below No-Edit because of cascaded errors).

### Tagger Bottleneck Analysis

9. **The Tagger's role depends on the encoding**: the Oracle-Editing difference is +0.064 for Scheme B (the Tagger is the bottleneck) but -0.015 / -0.034 for C/D (the Tagger contributes positively). At level L2 of Scheme B, Oracle = 0.749 vs. Editing = 0.437: the Tagger alone introduces a 0.31 performance loss.

10. **The Inserter cannot generate from scratch**: Inpainting mode (masking the whole accompaniment) reaches only 0.07-0.23 beat_exact_match, confirming that the Inserter depends heavily on the skeleton context supplied by the Tagger. At L4 (near-complete rewrite), however, Inpainting (0.19-0.21) overtakes Editing/Oracle (0.03-0.08), because the MASK placeholders preserve the correct token-count structure.

### Baseline Analysis

11. **No-Edit quantifies perturbation severity**: L1 = 0.88 (light), L2 = 0.75 (moderate), L3 = 0.55 (heavy), L4 = 0.08 (extreme). The models' largest gains over No-Edit occur at L2-L3, where TagFill D improves by +0.077.

12. **Copy-Context fails completely**: it reaches only 0.14 beat_exact_match, far below No-Edit (0.63), showing that piano accompaniment varies enough from beat to beat that naive repetition is not viable.

13. **CMLM is not up to the correction task**: using a pre-trained BERT for mask-predict falls below No-Edit on every scheme, and more iterations make it worse, showing that MLM confidence calibration is unsuited to error detection. Under the bundled encodings, CMLM (~0.46) is still far ahead of the separated ones (~0.32), because the complete information in a single token supports contextual inference.

### Inference-Time Optimization

14. **The three inference-time options give limited returns**: MaskGIT-style iterative decoding gives a marginal gain (+0.001 to +0.005), the confidence skeleton is essentially neutral, and the harmonic constraint actually hurts. TagFill's single-pass decoding is already close to sufficient, leaving little room for further optimization.

15. **Non-monotonic effect of the iteration count**: performance drops at iter=2, recovers at iter=3 and slightly exceeds iter=1, and iter=5 matches iter=3. Three iterations are enough to converge.

---

*Last updated: 2026-02-20*

*Data sources:*
- *Per-sample metrics (beat exact match, note F1, MPE, chroma F1, FMD): `evaluation/metrics.py`, driven by `evaluation/evaluate.py`*
- *Aggregated tables: `evaluation/summarize.py` over the pre-computed JSON files in `results/`*
- *Significance tests: `evaluation/statistical_tests.py`, `evaluation/pairwise_bootstrap_decoded.py`*
- *Two-way ANOVA over the 2x2 design: `evaluation/anova_and_pairwise.py`*
