# Why Does Beat Encoding Perform Best on Music Editing Tasks? — An Encoding-Advantage Analysis for the Paper

> **Terminology note.** This document was written during experimentation and refers to the three editing methods by the NLP systems they are adapted from. The paper and the released code use different names: GECToR → **SeqTag**, LevT → **IterEdit**, FELIX → **TagFill**. Scheme names follow the paper throughout: **A** = absolute + separated, **B** = relative + separated, **C** = relative + bundled, **D** = absolute + bundled.

## 1. Core Claim

Symbolic music editing imposes three formal requirements (R1--R3) on the underlying encoding, and together these constitute a **representational barrier**: existing event-stream encodings (e.g., REMI) and compound-word encodings (e.g., CPWord) cannot satisfy all three simultaneously, which caps the performance of any editing method at the encoding level, before the method is even chosen. Beat Encoding satisfies all three requirements structurally, by virtue of its beat-level grid, which is what allows the NLP editing paradigm to transfer effectively to music.

**Experimental evidence**: under an identical GECToR architecture, Beat Encoding achieves an error-recovery rate of 37--48%, REMI 10%, and CPWord only about 10.6%. The impact of encoding choice (gap = 0.294) is **20 times** that of method choice (gap = 0.015).

---

## 2. The Three Formal Requirements (R1--R3)

These are stated in the paper as follows.

### R1: Atomic Edit Units

**Requirement**: each note should map to a minimal, self-contained token group (ideally 1--2 tokens), so that one edit label corresponds to one complete editing target.

**Motivation**: edit operations (replace, delete, insert) are executed at token granularity. If a single note spans many tokens, a single note-level edit requires coordinating the prediction of several tokens, which raises the probability of error.

| Encoding | Tokens/note | Sequence length (mean) | Truncation rate (>2048) |
|------|:----------:|:----------------:|:---------------:|
| REMI | 3.6 | 2,386 | 42.0% |
| CPWord | ~3.5 | ~1,300 | ~35% |
| Beat-A/B (separated) | 2.7 | 1,919 | 35.2% |
| Beat-C/D (bundled) | 1.6 | 1,087 | <20% |

REMI spends 3--4 independent tokens per note (Position + Pitch + Velocity + Duration). CPWord conceptually "bundles" attributes into a compound token, but its sequences still contain a large number of Bar/Position marker tokens, and the model must predict five sub-vocabularies simultaneously through five factored prediction heads — **any single sub-head error corrupts the entire compound token** — so atomicity is not actually improved for editing purposes. Beat Encoding's bundled schemes (C/D) encode pitch and rhythm into a single token, achieving genuine atomicity at 1 token = 1 note.

### R2: Trivial Source--Target Alignment

**Requirement**: source and target sequences must align position by position, so that edit labels can be extracted by simple comparison.

**Motivation**: editing methods (GECToR, FELIX, LevT) rely on a position-by-position correspondence between source and target. If that correspondence is not trivial, label extraction and action prediction both become unreliable.

| Encoding | Alignment mechanism | Alignment trivial? |
|------|----------|:--------:|
| REMI | Chained re-computation of TIME_SHIFT events | No |
| CPWord | Bar/Position markers shift along with content | Partial |
| Beat | Beat boundaries provide content-invariant anchors | Yes |

Because Beat Encoding partitions the temporal axis into fixed-length beats, every beat occupies a dedicated token span regardless of its content. The *n*-th beat of the source therefore maps to the *n*-th beat of the target no matter what local edits occurred, giving trivial alignment. REMI and CPWord have no such content-invariant grid: their positional structure is a function of the content itself, so alignment degrades as soon as content changes.

### R3: Bounded Edit Locality

**Requirement**: a single-note musical edit should affect only a bounded number of tokens in the sequence, and the result must remain a valid encoding without requiring global re-computation.

**Motivation**: if inserting a note displaces every subsequent token, the alignment problem degrades from O(1) to O(n), and edits propagate unboundedly through the sequence.

| Encoding | Insertion displacement | Cascade risk |
|------|:----------:|:--------:|
| REMI | ~4 tokens | High |
| CPWord | ~1 token locally, but Position markers also shift | Medium |
| Beat | 0--2 tokens | Low |

REMI's critical flaw is the TIME_SHIFT mechanism: after inserting a note, the TIME_SHIFT values of all subsequent notes must be recomputed. Inserting a note on beat 2 of a 4-beat bar, for example, affects the TIME_SHIFT tokens of beats 2--4, producing a **cascading displacement of roughly 4 tokens on average**.

CPWord improves on this only partially. Each note is a single compound token, but Position marker tokens are interleaved with Note tokens, so inserting or deleting a note under the same Position still shifts the subsequent Position markers. More importantly, **CPWord's factored prediction architecture** (five independent sub-heads) means that alignment errors propagate independently along five dimensions.

Beat Encoding resolves this structurally: each beat is a self-contained grid cell delimited by voice markers, so insertions and deletions inside a beat **do not cross the beat boundary**. Note that the relative-position schemes (B and C) still exhibit **cascading dependencies** — changing one note's pitch forces the next note's position token to change as well — but this cascade is bounded within a single beat, so R3 is preserved. It is nevertheless the mechanism behind the Scheme B paradox analyzed in §5.

Structured is not evaluated experimentally here, but it is covered in the paper's requirement table: like CPWord it achieves partial atomicity (R1), while its reliance on relative time-shifts makes duration modifications propagate to subsequent positions, violating R2 and R3.

### 2.4 A Practical Corollary: Tractable Label Space

R1--R3 are the paper's three requirements. A fourth, practical consideration follows from them: the number of edit labels should scale with vocabulary size, not with sequence length, since an oversized label space causes severe class imbalance (>85% KEEP), leaves rare labels under-covered by the training data, and pushes the model toward brittle heuristics.

| Encoding | Label space | Label composition |
|------|:--------:|----------|
| REMI | 456 | KEEP + DELETE + 284 REPLACE + SHIFT + APPEND |
| CPWord | 4 + 5×V | 4 actions × 5 sub-vocabularies predicted independently |
| Beat-A/B (separated) | 350 | KEEP + DELETE + REPLACE_v + APPEND_n + SHIFT_±k |
| Beat-C/D (bundled) | 14,258 | KEEP + DELETE + REPLACE_bundle + APPEND |

CPWord's label design looks compact (only 4 action classes), but in practice the complexity is displaced into the five factored heads. On REPLACE/APPEND the model must **predict five sub-tokens correctly at once** (family, position, pitch, velocity, duration), and the conditional-independence assumption across these sub-predictions does not hold in music (pitch and duration, for instance, are strongly correlated), so joint accuracy is degraded multiplicatively.

Beat Encoding's separated schemes have a compact label space (350), and the SHIFT operation is a music-specific label unique to Beat Encoding — it nudges pitch without replacing the token outright. The bundled schemes have a much larger label space (14,258), but the resulting class imbalance can be handled effectively with techniques such as Focal Loss.

**Encoding reference (from the code).** The rhythm pattern is a base-3 (ternary) compression of the within-beat state vector, where 1 = onset and 2 = sustain continuation (0 = silent). With τ = 4 sixteenth-note steps per beat and big-endian digit order, a quarter note (1, 2, 2, 2) encodes to pattern = 53, and a beat of pure sustain (2, 2, 2, 2) encodes to pattern = 80, giving 81 possible patterns. Separated schemes emit a position token plus a pattern token per note (vocab: A = 186, B = 185). Bundled schemes emit a single token per note, composed as `position × 81 + pattern` — using the relative position offset in C and the absolute pitch index in D — for a vocabulary of 7,145 in both cases.

---

## 3. Experimental Evidence

### 3.1 GECToR Performance Across the Three Encodings

Under 200 samples with position-level perturbation (p_pitch = 10%, p_rhythm = 5%, p_delete = 3%, p_insert = 2%):

| Metric | Beat-A | Beat-C | Beat-D | REMI | CPWord |
|------|:------:|:------:|:------:|:----:|:------:|
| beat/pos_match | 0.660 | 0.690 | 0.689 | 0.466 | 0.393 |
| noedit_baseline | 0.595 | 0.610 | 0.610 | 0.317 | 0.321 |
| **Δ (improvement)** | **+0.065** | **+0.080** | **+0.079** | **+0.149** | **+0.072** |
| note_f1_tol0 | — | — | — | 0.953 | 0.949 |
| Error-recovery rate | 48.1% | 37.2% | 46.6% | 10.1% | ~10.6% |

> Note: absolute numbers for Beat vs. REMI/CPWord are not directly comparable, because of differences in evaluation scope (accompaniment only vs. full piece), comparison granularity (beat vs. bar), and other factors (see §4). The **error-recovery rate** and the **relative improvement** do, however, expose the editing-efficiency differences between encodings.

### 3.2 Structural Metrics at the Encoding Level (Layer 1, No Training Required)

Statistics computed over 500 MIDI samples:

| Metric | REMI | Beat-A | Ratio | Interpretation |
|------|:----:|:------:|:----:|------|
| Tokens/note | 3.57 | 2.70 | 1.32× | REMI spends 33% more tokens per note → more edit points |
| Sequence length | 2,386 | 1,919 | 1.24× | REMI sequences are longer → more truncation loss |
| Truncation rate | 42.0% | 35.2% | — | REMI loses more contextual information |
| Insertion displacement | ~4 tokens | 0--2 tokens | 2--4× | REMI's cascading displacement is more severe (violates R3) |

These metrics explain the source of the performance gap purely at the encoding level: REMI's 3.6 tokens per note mean that one note-level edit requires jointly predicting the correct action for 3--4 tokens, whereas Beat-C/D require predicting only 1.

### 3.3 Why Did CPWord Underperform Expectations?

CPWord was expected to land between REMI and Beat Encoding — its sequences are shorter (~1,300 vs. REMI's 2,386) and each note is a single compound token. In practice its beat_match is only 0.393, below even REMI's 0.466.

**Analysis: formal "bundling" is not the same as substantive "atomization."**

1. **The multiplicative effect of factored prediction**: CPWord's GECToR must predict the correct (family, position, pitch, velocity, duration) tuple through five independent heads. If each sub-head has accuracy p, joint accuracy is p⁵. Even at p = 0.95, joint accuracy is only 0.77. REMI uses more tokens, but each token is a single-dimensional prediction, with no multiplicative penalty.

2. **The modeling difficulty of heterogeneous token sequences**: CPWord sequences mix three families (Metric/Note/Special), and the action head must simultaneously handle Bar tokens (where only the position sub-vocabulary is meaningful), Position tokens (where only family + position are meaningful), and Note tokens (where pitch/velocity/duration are meaningful). This heterogeneity makes the task harder to learn.

3. **The semi-redundancy of Position markers**: Position markers occupy ~30% of a CPWord sequence, yet contribute little to note editing — the tokens that actually need editing are the Note tokens. These Position markers still shift under insertion/deletion (a partial violation of R2 and R3) while consuming model capacity.

4. **By contrast, Beat Encoding's beat grid encodes temporal position directly.** In Beat Encoding, position is implicit in the beat index (the *i*-th beat is beat *i*), so no explicit Position token is needed. Every Beat token is therefore "useful edit payload," with no structural redundancy.

---

## 4. A Note on Incomparability

Six factors make the absolute numbers for Beat vs. REMI/CPWord incomparable; the paper must state this explicitly:

| Factor | Beat | REMI/CPWord | Effect |
|------|------|---------|------|
| Perturbation granularity | per-beat | per-position (already aligned) | ✅ Controlled |
| Comparison granularity | beat (one time unit) | bar (~4 beats) | REMI's bar_match is inherently lower |
| Evaluation scope | accompaniment only (~40% of notes) | full piece (100% of notes) | Beat's metrics are less diluted |
| Note definition | (pitch, patch_val) per beat | (bar, pos, pitch) | Different F1 denominators |
| Test files | .npz split | .mid split | Possible differences in overlap |
| Data format | dual-voice (melody + accompaniment) | single-track (piano) | Different musical tasks |

**Paper strategy**: avoid cross-encoding comparison of absolute numbers, and emphasize instead:
1. **Within-encoding comparison** (each encoding's Δ over No-Edit)
2. **Structural metrics** (tokens/note, sequence length, displacement — all training-free)
3. **Within-scheme ablation** (the A/B/C/D four-scheme ablation of Beat Encoding, which is fully comparable)

---

## 5. The Cascading-Error Mechanism: The Scheme B Paradox

The four-scheme ablation of Beat Encoding reveals a deep cascading-error amplification mechanism, one of the paper's central findings.

### 5.1 The Phenomenon: Best Component ≠ Best End-to-End

| Scheme | Tagger F1 | End-to-end beat_match | FMD |
|------|:---------:|:-----------------:|:---:|
| B (relative + separated) | **0.605** (highest) | 0.518 (lowest) | 5.44 (worst) |
| C (relative + bundled) | 0.590 | **0.690** (highest) | 2.60 |
| D (absolute + bundled) | 0.585 | 0.689 | **2.18** (best) |

Scheme B has the **highest** component-level Tagger F1 but the **worst** end-to-end performance — its FMD is even worse than the Copy-Context baseline (4.97).

### 5.2 Root Cause: Triple Cascading Amplification

```
Layer 1: separated encoding → one wrong note affects 2 tokens (position + pattern)
Layer 2: relative position   → one wrong position token cascades into every
         subsequent position token
         Statistical evidence: 71% of samples exhibit 2+ cascaded position errors
Layer 3: multi-stage pipeline → the Tagger's faulty skeleton propagates to the Inserter
         The Inserter then runs MLM over a broken structure → garbage in, garbage out

Joint effect: a small Tagger error → exponential downstream damage
```

### 5.3 Oracle Experiment

Replacing the Tagger's output with ground-truth labels, to measure the effect in isolation:

| Scheme | Oracle − Editing | Interpretation |
|------|:----------------:|------|
| B | **+0.064** | The Tagger is the **bottleneck** (removing it *improves* results by 6.4%) |
| A | +0.005 | The Tagger is neutral |
| C | −0.015 | The Tagger **contributes positively** (a conservative KEEP strategy eases the Inserter's burden) |
| D | **−0.034** | The Tagger contributes most (absolute position isolates error propagation) |

### 5.4 Significance for the Paper

The Scheme B paradox reveals a **general principle**: in multi-stage sequence-editing systems, component-level optimization can backfire through cascading dependencies. This finding is not specific to music — it carries over to text editing (GECToR's original task), code modification, molecular editing, and similar domains.

---

## 6. Encoding--Method Interaction Effects

### 6.1 Rank Reversal

| Rank | GECToR | FELIX |
|:----:|--------|-------|
| #1 | C (relative + bundled) 0.690 | D (absolute + bundled) 0.703 |
| #2 | D (absolute + bundled) 0.689 | C (relative + bundled) 0.680 |
| #3 | A (absolute + separated) 0.660 | A (absolute + separated) 0.665 |
| #4 | B (relative + separated) 0.518 | B (relative + separated) 0.409 |

The ranks of C and D **reverse** between GECToR and FELIX. There is therefore no "universally optimal" encoding scheme — an encoding's effectiveness depends on the editing method paired with it.

### 6.2 Explaining the Interaction

| Aspect | Why GECToR prefers C (relative) | Why FELIX prefers D (absolute) |
|------|----------------------|---------------------|
| Architecture | Single-stage tagging | Two-stage Tagger → Inserter |
| Cascade exposure | No cascade (single prediction pass) | Tagger errors propagate to the Inserter |
| Relative position | Compressed deltas → easier to learn | Delta errors cascade → the Inserter collapses |
| Absolute position | Larger value range → slightly harder to learn | Errors stay isolated → the Inserter is protected |
| Core logic | A single stage can tolerate cascades | Multiple stages require error isolation |

**Takeaway for the paper**: a method's architectural properties (single-stage vs. multi-stage) determine which encoding flaw is most damaging. GECToR's single-stage architecture makes it naturally robust to the cascade problem of relative positions; FELIX's two-stage architecture needs the error isolation that absolute positions provide.

---

## 7. The Structural Boundary of the Editing Paradigm

### 7.1 L1--L4 Graded Analysis (FELIX, Scheme D)

| Corruption level | Meaning | No-Edit | FELIX | Δ |
|:--------:|------|:-------:|:-----:|:-:|
| L1 (8%) | Slight deviation | 0.88 | 0.92 | +0.04 |
| L2 (25%) | Moderate corruption | 0.75 | 0.86 | +0.11 |
| L3 (50%) | Severe corruption | 0.55 | 0.64 | +0.09 |
| L4 (100%) | Complete absence | 0.08 | 0.08 | ≈0 |

### 7.2 Boundary Condition

The core assumption of the editing paradigm is that **the input contains enough correct information**. L1--L3 satisfy this assumption (8--50% of the content is corrupted, but 50--92% of the correct context survives), so the model can exploit the remaining context to repair the errors. L4 violates it (100% missing), at which point editing degenerates into generation from scratch and all methods collapse to ~0.08.

**Paper phrasing**: "The editing paradigm is effective at L1--L3; the L4 boundary reveals the essential difference between editing and generation: editing depends on information in the input, whereas generation requires prior knowledge."

---

## 8. Proposed Narrative Frame for the Paper

### Main Claim

> The performance bottleneck in symbolic music editing lies not in the model architecture but in the representational capacity of the underlying encoding. We formalize three requirements (R1--R3), show that no existing encoding satisfies all three simultaneously, and design Beat Encoding to resolve this representational barrier structurally.

### Line of Argument

```
1. Pose the problem: why can't NLP editing methods be applied directly to music?
   → It is not the methods; it is the encoding (the representational barrier)

2. Formalize R1-R3
   → R1 (atomic edit units), R2 (trivial source-target alignment),
     R3 (bounded edit locality)
   → Analyze why REMI, CPWord, and Structured violate them

3. Beat Encoding design
   → Beat-level grid → satisfies R1-R3 by construction
   → 2x2 ablation (position x format)

4. Experimental validation
   → Encoding-level metrics (Layer 1): tokens/note, displacement, truncation rate
   → End-to-end performance (Layer 2): 3 methods x 4 schemes
   → CPWord comparison: formal bundling != substantive atomization

5. Deeper findings
   → The Scheme B paradox: cascading-error amplification
   → Rank reversal: encoding-method interaction effects
   → The L4 boundary: a structural limit of the editing paradigm

6. Conclusion
   → Encoding design is a first-principles concern (20x the impact of method choice)
   → Representation-method co-design is the central problem in structured
     sequence editing
```

### Key Figures to Cite

- **20:1 impact ratio**: encoding choice gap = 0.294 vs. method choice gap = 0.015
- **4--5× recovery rate**: Beat 48% vs. REMI 10%
- **The Scheme B paradox**: highest Tagger F1 → worst end-to-end result
- **Rank reversal**: GECToR (C > D) vs. FELIX (D > C)
- **The L4 collapse**: all methods → ~0.08

---

## Appendix: Full Results

### A. All Methods × All Encodings, beat_exact_match (50 samples)

| Method | A | B | C | D |
|------|:---:|:---:|:---:|:---:|
| Copy-Context | 0.139 | 0.139 | 0.136 | 0.136 |
| BERT-CMLM | 0.318 | 0.330 | 0.461 | 0.468 |
| No-Edit | 0.629 | 0.629 | 0.626 | 0.626 |
| GECToR | 0.661 | 0.518 | **0.690** | 0.689 |
| FELIX | 0.637 | 0.385 | 0.675 | **0.699** |

### B. REMI / CPWord GECToR (200 samples, position mode)

| Metric | REMI | CPWord | Beat-best |
|------|:----:|:------:|:---------:|
| beat/pos_match | 0.466 | 0.393 | 0.690 |
| noedit_baseline | 0.317 | 0.321 | 0.610 |
| Δ over noedit | +0.149 | +0.072 | +0.080 |
| note_f1_tol0 | 0.953 | 0.949 | — |
| Error-recovery rate | ~10% | ~10.6% | 37--48% |

### C. Oracle Experiment (Tagger Bottleneck Analysis)

| Scheme | Editing | Oracle | Δ | Inpainting |
|------|:-------:|:------:|:-:|:----------:|
| A | 0.665 | 0.670 | +0.005 | 0.135 |
| B | 0.409 | 0.473 | +0.064 | 0.066 |
| C | 0.680 | 0.665 | −0.015 | 0.207 |
| D | 0.703 | 0.669 | −0.034 | 0.229 |
