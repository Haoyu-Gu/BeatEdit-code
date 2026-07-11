# Method: Levenshtein-Transformer-Based Symbolic Music Editing (IterEdit)

> Design notes for **IterEdit** (paper §3.3), the iterative deletion–insertion–prediction
> mechanism of BeatEdit. This document focuses on the rationale behind the design
> choices; it is written as background for the paper and for readers of the code.
>
> Where a design discussed here was explored during development but did not make it
> into the released model (or was measured and found harmful), this is stated
> explicitly in a **Status** note. The released implementation lives in
> `src/iteredit/`.

---

## 1. Motivation

### 1.1 Why symbolic music inpainting?

Symbolic music inpainting means: given a MIDI sequence in which some region is masked
out (e.g. 4–8 bars of melody are missing), the model must generate a completion that is
stylistically coherent with the surrounding context and structurally sound.

The task has direct value in real music creation:
- **Composition assistance**: the composer writes the opening and the ending, and the model fills the transition in between.
- **Score restoration**: recovering damaged or missing passages of a score.
- **Conditional generation**: generating an accompaniment under a given melodic skeleton.

Unlike unconditional generation, the core difficulty of inpainting is **bidirectional
consistency** — the generated fragment must join up with both the left and the right
context. This is a natural obstacle for purely autoregressive (AR) models, which can
only see the left context.

In BeatEdit, IterEdit is the mechanism for the **medium-to-high edit-density** regime:
accompaniment editing (its primary task), error correction, and segment completion.

### 1.2 Limitations of existing approaches

| Method family | Representative work | Limitation |
|---------------|--------------------|------------|
| Autoregressive (AR) | Music Transformer, MuseNet | Unidirectional generation; cannot use the right context; inpainting needs an extra infilling trick |
| Masked language models (MLM) | MusicBERT, Mask-Predict | Predicts all positions at once; cannot handle inpainting where the **length is unknown** (the source and target lengths differ) |
| Diffusion models | DiffuSeq, DDPM for music | Expensive; many iteration steps; the discrete symbolic domain needs extra quantization |
| Seq2Seq | Encoder–Decoder | Forces inpainting into a translation formulation; context is used unnaturally |

**The central tension**: the target sequence of a music inpainting task has an
**unknown length**. After masking 4 bars, the target region might be 50 tokens or 200 —
it depends on the note density of that region. Fixed-length BERT-style mask-predict
cannot handle this.

### 1.3 Why the Levenshtein Transformer?

The Levenshtein Transformer (Gu et al., 2019) was originally a non-autoregressive (NAR)
method for machine translation. Its core idea is to decompose sequence generation into
three edit policies:

$$\text{Delete} \rightarrow \text{Insert Placeholders} \rightarrow \text{Fill Tokens}$$

This **"generation as editing"** view fits music inpainting naturally:

1. **Length adaptivity**: the placeholder-insertion policy predicts how many placeholders to insert in each gap, inferring the target length automatically — no need to specify it in advance.
2. **Bidirectional context**: an encoder-only architecture gives global attention, so the left and right context are visible at once.
3. **Iterative correction**: several rounds of delete → insert → fill approach the target progressively, instead of gambling on predicting every token in one shot.
4. **Context protection**: with a constrained Levenshtein DP, the context region can be frozen so that only the masked region is edited.

**Our contribution**: porting the Levenshtein Transformer from NLP translation to
symbolic music editing, with systematic adaptations for the specifics of the music
domain.

---

## 2. Method Overview

The overall pipeline:

```
Complete MIDI sequence
      ↓
[context] ████masked region████ [context]   ← create the inpainting pair (beat-boundary masking)
      ↓
[context] [PLH] [context]                   ← initial state: a single placeholder
      ↓
┌────────────────────────────────────────┐
│  Iterative loop (max 10 rounds)         │
│  ① Deletion:    delete wrong tokens     │
│  ② Insertion:   insert PLH into gaps    │
│  ③ Token pred.: fill PLH with real tok. │
│  Convergence: no edit of any kind → stop│
└────────────────────────────────────────┘
      ↓
[context] [generated fragment] [context]    ← final output
```

**Summary of the design contributions:**

| Contribution | Description |
|--------------|-------------|
| **Constrained Levenshtein DP** | The edit distance is computed only inside the masked region; the context is frozen and takes no part in the DP |
| **Beat-boundary masking** | Masking follows musical beat boundaries instead of random token positions |
| **Per-token editable flags** | Boolean flags replace fragile index-based boundary tracking, so the editable region is maintained robustly during inference |
| **2×2 encoding-scheme study** | A systematic comparison of absolute/relative × separated/bundled music representations |
| **Music BERT pre-trained initialization** | Warm-starting the encoder from music-domain MLM pre-training |
| **Multi-level perturbation training** | Four perturbation levels (L1–L4) plus intermediate-state sampling, so the model generalizes beyond contiguous masking (§5) |

---

## 3. Music Representation: the Four Encoding Schemes

### 3.1 Design rationale

Symbolic music (MIDI) is essentially a **two-dimensional piano roll** (pitch × time) and
has to be serialized into a one-dimensional token sequence before it can be fed to a
Transformer. The serialization directly determines:
- the vocabulary size (which affects how hard prediction is),
- the sequence length (which affects the attention cost),
- whether the representation is translation-invariant (which affects generalization).

We use a **2×2 factorial design**, taking two options along each of two independent axes:

|  | Separated (unbundled) | Bundled |
|--|-----------------------|---------|
| **Absolute position** | Scheme A (vocab = 186) | Scheme D (vocab = 7145) |
| **Relative position** | Scheme B (vocab = 185) | Scheme C (vocab = 7145) |

The vocabulary sizes above are the model vocabularies of the shared Music BERT backbone
(base vocabulary + `[MASK]`). IterEdit adds one more symbol, the `[PLH]` placeholder, so
its own vocabulary is 7146 for Schemes C/D (see `src/iteredit/configs/config.py`, whose
constants are stated for Scheme C, and `SCHEME_TOKENS` in
`src/iteredit/data/dataset_editing.py`, which covers all four schemes).

### 3.2 The pitch-encoding primitive: the ternary patch

All four schemes share the same **note-state encoding**. The piano roll is cut into
patches of `patch_h = 1` pitch × `patch_w = 4` time steps, and each patch is encoded as a
4-digit ternary number — one digit per time step:

```
patch = [d0, d1, d2, d3]           # four consecutive time steps at one pitch
each digit ∈ {0, 1, 2}:  0 = silent, 1 = onset, 2 = sustain continuation
patch_value = d[0]×27 + d[1]×9 + d[2]×3 + d[3]×1  ∈ [0, 80]
```

There are $3^4 = 81$ patch values, and these are the basic unit in which note content is
expressed. Two examples, to fix the convention:

- a quarter note (an onset held for the whole beat) is `(1,2,2,2)` = **53**;
- a beat of pure sustain, with no new attack, is `(2,2,2,2)` = **80**.

### 3.3 Axis one: relative vs. absolute position encoding

**Relative (Schemes B/C):**
```
position of token[i] = position of token[i-1] + relative_distance[i]
```
- Advantage: **translation invariance** — transposing a phrase leaves the relative encoding unchanged, which helps generalization.
- Advantage: in sparse passages (large pitch gaps between notes) the position values are smaller and more concentrated.
- Drawback: decoding requires a **cumulative sum**, so a single positional error cascades to everything that follows.

**Absolute (Schemes A/D):**
```
position of token[i] = the pitch index directly, ∈ [0, 87]
```
- Advantage: **every token is independently decodable**; there is no error propagation.
- Advantage: better suited to parallel decoding.
- Drawback: translation invariance is lost — a transposed passage encodes completely differently.

**Hypothesis (and outcome).** We expected relative encoding's translation invariance to
help generalization, at the risk that its cumulative error would be amplified across
iterations. The measured result confirms the risk and not the benefit: for IterEdit,
absolute encoding is consistently better, precisely because relative encoding's
cascading dependencies compound across refinement rounds (paper §5.3).

### 3.4 Axis two: separated vs. bundled token organization

**Separated (Schemes A/B).** Each note is represented by **two tokens** — a position
token plus a content token:
```
[81 + position, patch_value]    → 2 tokens/note
vocabulary: 81 (patch) + 88 (position) + special tokens ≈ 186
```

**Bundled (Schemes C/D).** Each note is represented by **one token**, with position and
content encoded jointly:
```
bundled = position × 81 + patch_value  → 1 token/note
vocabulary: 88 × 81 + special tokens = 7145
```

**The core trade-off:**

| Criterion | Separated (A/B) | Bundled (C/D) |
|-----------|-----------------|---------------|
| Vocabulary size | ~186 (small) | ~7145 (large) |
| Sequence length | 2 × #notes (long) | 1 × #notes (short) |
| Token-prediction difficulty | Low (186-way) | High (7145-way) |
| Use of the context window | Worse (long sequences) | Better (short sequences) |
| Embedding parameters | ~95 K | ~3.7 M |

**Hypothesis**: bundled sequences are shorter, so the Transformer sees a larger musical
context; but the vocabulary explodes from 186 to 7145, making the token-prediction
policy roughly 38× harder as a classification problem.

### 3.5 A concrete encoding example

Suppose some beat has 3 notes at pitches [10, 35, 60] with patch values [5, 23, 41]:

| Scheme | Encoding | #Tokens |
|--------|----------|---------|
| A (abs/separated) | `[TRACK0, 91, 5, 116, 23, 141, 41]` | 7 |
| B (rel/separated) | `[91, 5, 106, 23, 106, 41, END]` | 7 |
| C (rel/bundled) | `[SPLIT_0, 815, 2048, 2066]` | 4 |
| D (abs/bundled) | `[SPLIT_0, 815, 2858, 4901]` | 4 |

Bundling saves about **43%** of the tokens in this example. Measured over the corpus, the
bundled schemes average 1,163 tokens per sequence against 1,891–1,907 for the separated
schemes (≈39% shorter).

### 3.6 Dual-track representation

The input is a dual-track piano roll (melody = Track 0, accompaniment = Track 1). Within
each beat the two voices alternate:

```
[BAR] [SPLIT_0] melody_notes... [SPLIT_1] accomp_notes... [BAR] ...
```

`SPLIT_0` / `SPLIT_1` (or `TRACK0_START` / `TRACK1_START`) mark the voice switch, so the
model can tell melody from accompaniment and model them separately. Because the
interleaving is at beat level and positional, separating the accompaniment out of the
sequence is a matter of positional indexing.

---

## 4. Architecture: Three Policy Heads on a Shared Encoder

### 4.1 Why encoder-only rather than encoder–decoder?

The original LevT (Gu et al., 2019) uses an encoder–decoder architecture, because in
translation the source and target are different sequences. But music editing is
fundamentally **local editing within a single sequence** — the context and the region
being repaired share one token space and one attention mechanism.

Advantages of encoder-only:
1. **Consistent with BERT pre-training**: the Music BERT encoder weights can be reused directly, with no cross-attention to initialize.
2. **Global attention**: every position sees the left and the right context at once, which is exactly what editing needs.
3. **Parameter-efficient**: roughly half the parameters of an encoder–decoder of the same width (33.9 M vs. 66.5 M), since there is no decoder and no cross-attention.
4. **Simplicity**: the three heads branch straight off the shared hidden states; there is no autoregressive decoder mask.

This is not merely a convenience. The paper's ablation (Appendix) replaces the
encoder-only backbone with an encoder–decoder cross-attention design and performance
**collapses** (editing beat exact match .051, correction .034, FMD > 18). The encoder-only
inductive bias — edited tokens and context tokens sharing one attention — is a
load-bearing part of the editing paradigm, not an optimization.

### 4.2 Architecture details

```
Input → BERT embeddings (token + learned position) → BertModel encoder (8L, 512H, 8 heads, 2048 FFN)
                                                ↙          ↓          ↘
                                           Del Head    Ins Head    Tok Head
                                           (B,L,2)   (B,L+1,21)  (B,L,V)
```

The backbone is a HuggingFace `BertModel` (`add_pooling_layer=False`), configured to
match the Music BERT pre-training exactly so that the pre-trained weights load directly
(`src/iteredit/models/levenshtein_transformer.py`).

| Component | Setting | Rationale |
|-----------|---------|-----------|
| Hidden dim | 512 | Matches Music BERT, so weights transfer |
| Layers | 8 | ~33.9 M parameters total |
| Attention heads | 8 | 64 dims per head; different heads cover musical dependencies at different ranges |
| FFN dim | 2048 | 4× hidden, the standard Transformer ratio |
| Activation | GELU | Smoother than ReLU; the BERT default |
| Norm | Post-LayerNorm (BERT default) | Inherited from the pre-trained backbone |
| Position encoding | Learned absolute embeddings, `max_position_embeddings = 2048` | Inherited from the pre-trained backbone |
| Dropout | 0.1 | Hidden and attention dropout |

### 4.3 The three policy heads

**Deletion — "which tokens are wrong?"**
```
LayerNorm(512) → Dropout → Linear(512, 2)
```
- Binary classification per token: KEEP (0) / DELETE (1).
- Deliberately the simplest head — deciding whether to delete is the easiest of the three judgements and does not need extra capacity.

**Placeholder insertion — "how many placeholders belong in each gap?"**
```
[h_{i-1}; h_i] → LayerNorm(1024) → Dropout → Linear(1024, 21)
```
- **Key design point**: adjacent hidden states are concatenated to build a "gap representation".
- A sequence of length L has L+1 gaps (including the two ends), so the output is `(B, L+1, 21)`.
- The prediction range is 0–20: at most 20 placeholders per gap (`max_insert = 20`).

**Why concatenate adjacent representations?** How many tokens belong in a gap depends on
how discontinuous the two sides are. Concatenating the hidden states of the neighbouring
tokens encodes that local discontinuity directly, which is sharper than using one side
alone or a global pooled vector.

**Token prediction — "what should each placeholder become?"**
```
Linear(512, 512) → GELU → LayerNorm → Linear(512, V)
```
- Runs on the sequence *after* placeholders have been inserted, predicting a target token for every position (the placeholder positions in particular).
- Uses a two-layer MLP rather than a single linear layer, because token prediction is the hardest of the three sub-tasks.

All three heads are initialized with Xavier-uniform weights and zero biases; the encoder
is initialized from Music BERT.

### 4.4 The two forward passes

Training (`compute_loss`) requires two independent encoder forward passes:

| Forward | Input sequence | Output | Why |
|---------|----------------|--------|-----|
| Pass 1 | $z$ (the intermediate state) | del_logits + ins_logits | Decide what to delete and where to insert, on the *current* state |
| Pass 2 | $z_{tok}$ ($z$ after placeholders are inserted) | tok_logits | Predict the fill on the state that *already contains* the placeholders |

The two passes **cannot be merged**, because $z_{tok}$ is longer than $z$ (extra `[PLH]`
tokens have been inserted) and is therefore a different sequence. This is an inherent
cost of the Levenshtein framework, and it buys the length adaptivity.

### 4.5 Music BERT pre-trained initialization

We pre-train a Music BERT (MLM objective) for each of the four encoding schemes, then
transfer the embedding layer and the encoder layers into the corresponding modules of
IterEdit. The three heads are initialized randomly (Xavier uniform).

**Why the transfer works:**
- BERT has already learned the local dependencies between music tokens (chord structure, rhythmic patterns, and so on).
- IterEdit's encoder is architecturally identical to the BERT encoder, so the weights load directly.
- The three heads are new tasks; random initialization does not hurt convergence (empirically their losses drop quickly within the first ~3 epochs).

The value of this initialization is large: in the paper's pre-training ablation, removing
BERT pre-training and training from scratch degrades performance catastrophically
(SeqTag correction beat 0.700 → 0.195).

---

## 5. Training

### 5.1 Constrained Levenshtein DP (core design point)

The standard Levenshtein distance is the minimum number of edits that turn sequence A
into sequence B. **Our key modification**: the DP is run only inside the masked region
$[m_s, m_e)$, and the labels for the context region are forced to "no operation".

```
full sequence:  [context_left | mask_region | context_right]
DP range:                      |←── this region only ──→|
context labels:  del = 0, ins = 0 (frozen)
```

The procedure:
1. Extract the masked region $z[m_s:m_e]$ of the intermediate state $z$, and the corresponding region of the target sequence $y$.
2. Run a standard Levenshtein DP between those two sub-sequences only.
3. Backtrace the DP path and read off three sets of labels:
   - `del_labels[i]` ∈ {0, 1}: delete or not;
   - `ins_labels[i]` ∈ [0, 20]: how many placeholders go in gap $i$;
   - `tok_labels`: the target token for each placeholder position.
4. Context positions get `ignore_index = -100` (they contribute no loss).

**Why not run the DP over the whole sequence?** If the DP were allowed to operate on the
context, it could "discover" paths that reduce the total edit distance by modifying the
context — which violates the task constraint that the context is immutable. The
constrained DP keeps the training labels consistent with the behaviour at inference.

**Implementation note.** `src/iteredit/data/levenshtein_utils.py` is a numpy
re-implementation of the alignment routines, ported from fairseq's
`fairseq/models/nat/levenshtein_utils.py`. It is a rewrite, not a wrapper: **fairseq is
not a runtime dependency** of this repository.

### 5.2 Intermediate-state sampling (the roll-in policy)

At inference the model sees inputs that are "halfway through being repaired" — some
tokens are already correct and some are still wrong. Training must simulate such
intermediate states in order to produce training pairs.

Our sampling policy is applied independently to every token of the masked region:
- **delete with probability 0.30**: simulates a leftover token from an earlier iteration;
- **replace with a random token with probability 0.20**: simulates a token an earlier iteration predicted wrongly;
- **keep the correct token with probability 0.50**: provides partially-correct information as an anchor.

**Rationale:**
- The 0.50 keep rate ensures the model learns to exploit the correct tokens it already has, rather than starting from nothing.
- The 0.30 delete rate simulates "there is redundant material in the sequence that must be cleaned up".
- The 0.20 replace rate simulates "there is a wrong token that must first be deleted and then regenerated".
- Together these three corruptions cover the kinds of intermediate state that actually arise during iterative refinement.

**This is deliberately not the original LevT roll-in.** The original Levenshtein
Transformer trains its policies with dual-policy imitation learning (roll-in from a
mixture of the model's own predictions and an expert/oracle policy). IterEdit instead
constructs the roll-in states by direct random corruption of the target's mask region, as
described above. The corruption probabilities are the
`corruption_delete_prob` / `corruption_replace_prob` fields in
`src/iteredit/configs/config.py`.

### 5.3 Multi-level perturbation (the editing task)

Contiguous masking alone teaches the model to fill a gap; it does not teach it to make
scattered, fine-grained edits. For the accompaniment-editing task, the training data is
therefore built by perturbing the **accompaniment track only** (the melody is left intact
as context) at one of four difficulty levels, sampled with weights
**L1 : L2 : L3 : L4 = 30 : 30 : 25 : 15** — from light single-note edits (L1) up to
near-complete rewriting of the accompaniment (L4). See
`src/iteredit/data/dataset_editing.py` and `perturb_accompaniment()`.

This matters a great deal: the paper's ablation shows a roughly 4× gap on editing beat
exact match between the multi-level-perturbation model (.480) and the vanilla
contiguous-masking configuration (.114).

### 5.4 Beat-boundary masking (the musical prior)

The masking strategy used when building inpainting/completion pairs:

1. Parse the full sequence into a structured list of beats along **beat boundaries**.
2. Choose a random run of **consecutive beats** as the masked region (12.5%–50% of the beats).
3. Keep at least one beat of context at each end.

**Why beat boundaries rather than random token positions?**

Music is hierarchical: notes → beats → bars → phrases. Masking random tokens can cut a
beat in half and produce unnatural boundaries (for instance, half of a chord). Masking on
beat boundaries guarantees that:
- the context has an intact beat structure, giving the model a clean musical prior;
- the target region has an intact beat structure, so the training labels are semantically well-defined;
- the setup matches the real use case (users specify a region to repair in bars or phrases).

### 5.5 Loss function

$$\mathcal{L} = \mathcal{L}_{del} + \mathcal{L}_{ins} + \mathcal{L}_{tok}$$

| Term | Formula | Notes |
|------|---------|-------|
| $\mathcal{L}_{del}$ | `CE(del_logits, del_labels, ignore=-100)` | Binary CE; context positions ignored |
| $\mathcal{L}_{ins}$ | `CE(ins_logits, ins_labels, ignore=-100)` | 21-way CE; only gaps in the masked region contribute |
| $\mathcal{L}_{tok}$ | `CE(tok_logits[PLH], tok_targets, ε=0.1)` | Placeholder positions only; label smoothing encourages diversity |

The three terms are equally weighted ($w_{del} = w_{ins} = w_{tok} = 1.0$). Empirically:
- the deletion and insertion losses fall to very small values (< 0.1) within the first few epochs, confirming that these are the easier sub-tasks;
- the token loss accounts for **95%+** of the final total loss — token prediction is the real difficulty;
- dynamic loss weighting, to send more gradient to the token head, remains an open direction.

### 5.6 Training hyperparameters

| Setting | Value |
|---------|-------|
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 3 × 10⁻⁴, cosine schedule with 10% warmup |
| Batch size | **64 effective** (32 per step × 2 gradient-accumulation steps) |
| Epochs | 30 |
| Mixed precision | fp16 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.1 (token head) |
| Max sequence length | 2048 |
| Mask ratio (beats) | 0.125 – 0.50 |

See `src/iteredit/configs/config.py` and `scripts/04_train_iteredit.sh`.

---

## 6. Iterative Inference: the Editable-Flag Mechanism

### 6.1 The three-stage loop

```python
initial state: [context(frozen)] + [PLH] + [context(frozen)]

for step in 1..max_iter:
    ① Delete: for tokens in the editable region, delete if p > 0.5
    ② Insert: for editable gaps, predict how many PLH to insert
    ③ Fill:   for every PLH position, predict the target token

    if no deletion + no insertion + no fill → converged, stop
```

### 6.2 Per-token editable flags (core design point)

This is our main change to the original LevT inference procedure. The original tracks the
editable region with index-based `mask_start` / `mask_end` boundaries, which is extremely
fragile under iterative editing:

**The problem**: every deletion or insertion changes the sequence length and shifts the
indices. For example:
- after deleting 3 tokens at position 5, a `mask_end` of 20 must become 17;
- after inserting 5 placeholders at position 10, `mask_end` must become 22;
- once several operations compose, index tracking is very easy to get wrong.

**Our solution**: maintain a boolean flag `editable[i]` for every token:
- `True`: editable (belongs to the masked region, or is a newly inserted token);
- `False`: frozen (belongs to the context; cannot be deleted or modified).

**The rules:**

| Operation | How `editable` is updated |
|-----------|---------------------------|
| Delete token $i$ | Remove `editable[i]`; the remaining flags are unchanged |
| Insert a PLH into gap $i$ | The new PLH gets `editable = True` |
| Fill a PLH with a real token | Keep `editable = True` (it can still be deleted in a later iteration) |
| Context token | Always `editable = False`; never deleted |

**Deciding whether a gap can take an insertion:**
```
gap[i] is insertable ⟺ editable[i-1] = True or editable[i] = True
```
That is: insertion is allowed only at the boundary of, or inside, the editable region.
This prevents the model from conjuring tokens into the middle of the context.

**Why this is the better design:**
1. **Decoupled from length changes**: the flag travels with its token, so insertions and deletions stay consistent automatically.
2. **Supports non-contiguous editable regions**: in principle several disjoint masked regions can be handled — which is exactly what the accompaniment-editing task needs, since the perturbed beats are scattered.
3. **Simple to implement**: Python list insertion/removal maintains the correspondence between tokens and flags for free.

This mechanism is also what enforces **hard melody preservation** in the editing task:
all melody tokens and structural markers are given $E_i = 0$, and the model is
constrained never to touch those positions — at training time (the frozen tokens are
identical in the intermediate state and the target, so the DP naturally emits `del = 0`,
`ins = 0` for them) and at inference time (the deletion head only operates where
$E_i = 1$).

### 6.3 The insertion seed on the first iteration

A special case: on the first iteration the masked region may contain only a single
placeholder, so `editable = [False, ..., True, ..., False]`. To make sure enough
placeholders can be inserted on the very first round, an `insertion_seed` is placed at the
mask-start position — that gap is allowed to take insertions even though the tokens on
both sides of it are frozen.

This is an engineering detail, but it is essential for correctness: without the insertion
seed, the first round could only operate at the single existing placeholder and could
never grow to the target length.

### 6.4 Convergence and termination

The convergence test is simple and effective:
- zero tokens deleted **and** zero placeholders inserted **and** zero placeholders filled → stop;
- or `max_iter` is reached.

In the released configuration `max_iter = 10` for the completion task (also the default
in `configs/config.py` and `evaluation/evaluate.py`), and **3 iterations** for the editing
task, where the input is already a plausible draft rather than an empty gap. The deletion
threshold is 0.5.

Semantically, "no operation at all" means the model considers the current sequence
optimal. In practice most completion samples converge within 3–5 rounds.

---

## 7. Design-Decision Analysis

### 7.1 Why non-autoregressive (NAR) rather than autoregressive (AR)?

| Dimension | AR | NAR (LevT / IterEdit) |
|-----------|-----|----------------------|
| Context | Left only | Bidirectional, global |
| Output length | Must be fixed in advance, or ended with EOS | Inferred dynamically by the insertion head |
| Inference cost | $O(n)$ sequential decoding | $O(k)$ iterations ($k \ll n$) |
| Error accumulation | Severe (each step depends on the last) | Mitigated by iterative correction |
| Diversity | Needs beam search or sampling | Temperature / top-k supported naturally |

### 7.2 Encoder-only vs. encoder–decoder

The original LevT paper uses an encoder–decoder because in translation source ≠ target.
In music editing, however:
- the source (the sequence with the masked/perturbed region) and the target (the complete sequence) share one token space;
- the context tokens are literally identical in source and target;
- there is nothing to "translate", so cross-attention has no job to do.

Encoder-only is therefore both more natural and more efficient. This was originally noted
as a hypothesis with encoder–decoder held in reserve as a fallback if capacity turned out
to be insufficient. **The fallback is closed**: the measured encoder–decoder variant
(66.5 M parameters vs. 33.9 M) does not merely fail to help, it collapses — editing beat
exact match .051 and correction .034, with FMD above 18.

### 7.3 Training–inference consistency

| Aspect | Training | Inference | Consistent? |
|--------|----------|-----------|-------------|
| Context / melody protection | Frozen tokens are identical in the intermediate state and the target, so the DP emits `del=0, ins=0`; context labels are additionally masked out of the loss (`ignore = -100`) | Hard constraint: `editable` flags freeze the tokens | Yes for the editing task; for the contiguous-masking mode the training-side constraint is a loss mask rather than a hard freeze |
| Source of the intermediate state | Random corruption (0.3 / 0.2 / 0.5) plus multi-level perturbation | The model's own output from the previous round | Gap remains |
| Edit sequence | The DP-optimal path | The model's greedy prediction | Broadly consistent |

**Acknowledged limitation**: for the contiguous-masking (completion) configuration, the
soft constraint used at training (ignoring the context gradient through a loss mask) does
not exactly match the hard constraint used at inference (freezing the tokens outright).
Intermediate-state sampling narrows, but does not close, the exposure-bias gap between
random roll-in states and the model's own multi-round outputs; a DAgger-style roll-in
remains an open direction.

### 7.4 Why the four-scheme comparison is a sound experimental design

The 2×2 factorial design lets us assess the two axes **independently**:
- fix the token organization and compare A vs. B (absolute vs. relative, separated);
- fix the token organization and compare D vs. C (absolute vs. relative, bundled);
- fix the position encoding and compare A vs. D (separated vs. bundled, absolute);
- fix the position encoding and compare B vs. C (separated vs. bundled, relative).

All four schemes share exactly the same model architecture, training hyperparameters,
dataset, and evaluation pipeline; the encoding is the only variable. That is what makes
the comparison fair — and it is what allows the encoding × method interaction effect to
be measured.

---

## 8. Relation to Prior Work

### 8.1 Differences from the original LevT (Gu et al., 2019)

| Dimension | Original LevT | This work |
|-----------|---------------|-----------|
| Task | Machine translation | Symbolic music editing (conditional generation) |
| Architecture | Encoder–decoder | Encoder-only |
| Edit scope | The whole sequence is editable | Constrained: only the masked / perturbed region is editable |
| Boundary tracking | None (everything is editable) | Per-token editable flags |
| Roll-in policy | Dual-policy imitation learning | Intermediate-state sampling by random corruption (§5.2) + multi-level perturbation (§5.3) |
| Pre-training | None | Music BERT initialization |
| Input representation | BPE subwords | Music-specific encodings (four schemes) |

### 8.2 Differences from Mask-Predict (Ghazvininejad et al., 2019)

Mask-Predict is also an iterative NAR method, but:
- it assumes the target length is known (predicted from the source length), so it **cannot handle variable length**;
- each round only replaces low-confidence tokens; it **cannot delete or insert**;
- the delete → insert → fill decomposition used here is strictly more flexible.

### 8.3 Position within music AI

| Method | Generation style | Length handling | Context |
|--------|------------------|-----------------|---------|
| Music Transformer (Huang et al.) | AR | Fixed or EOS | Left only |
| MuseNet (OpenAI) | AR | Fixed | Left only |
| MusicBERT + Mask-Predict | NAR | Fixed | Bidirectional |
| **IterEdit (ours)** | **NAR, iterative** | **Adaptive** | **Bidirectional** |

The distinctive claim is length-adaptive bidirectional editing for symbolic music.

---

## 9. Track-Coordinated Editing (explored)

> **Status.** This section records a line of work that was explored during development.
> The **track-aware attention bias (§9.2) was measured and it hurts** — it is not part of
> the released model, and `src/iteredit/training/*.py` explicitly strips any `track_bias`
> tensor from a loaded checkpoint. The **conditional track editing idea (§9.3) did ship**,
> but in a different form from the one sketched below: rather than a `melody_only` /
> `accomp_only` masking mode, the released code realizes it through dedicated datasets
> (`data/dataset_editing.py`, `data/dataset_accomp_inpainting.py`) that perturb or mask
> the accompaniment while keeping the melody as frozen context.

### 9.1 Motivation: modelling voice relationships explicitly

In a dual-track piece the tokens of both voices sit in one sequence, and self-attention
treats them all alike. But music has two fundamentally different kinds of dependency:

- **Within-voice**: the melodic line must flow; the accompaniment figuration must stay consistent.
- **Cross-voice**: harmonic compatibility (no clashing intervals), rhythmic complementarity (the accompaniment thins out where the melody is dense).

A standard Transformer has to learn the difference between these implicitly from data.
Since the encoding already carries voice information (the `SPLIT_0` / `SPLIT_1` markers),
the idea was to inject that structural prior into attention explicitly.

### 9.2 Track-aware attention bias — measured, and rejected

**Method.** Add a learnable voice-relationship bias to the attention scores:

$$\text{attn}(h, i, j) \mathrel{+}= \text{track\_bias}[h, \text{track}(i), \text{track}(j)]$$

Each token is assigned `track_id ∈ {0 = structural, 1 = melody, 2 = accompaniment}`. Each
of the 8 attention heads learns its own 3×3 bias matrix, for 72 parameters in total
(+0.0002%). The bias is injected as an additive float tensor of shape `(B × nH, L, L)`
through the `mask` argument of PyTorch's `TransformerEncoder`, so no attention-layer code
has to change. Initializing at zero makes it a no-op at the start of training, leaving the
pre-trained weights undisturbed.

**Result.** It degrades performance on both tasks: on Scheme A, editing beat exact match
falls from .448 to .428 (−4.5%) and correction from .719 to .666 (−7.4%). The conclusion
reported in the paper is that shared attention already captures the inter-track
dependencies, and the explicit structural bias adds nothing but constraint. The released
model does not include it.

The alternatives that were considered and not taken, recorded for completeness:

| Alternative | Why not |
|-------------|---------|
| Add a track embedding to the hidden state | Permanently changes the representation space; a bias only modulates "who attends to whom", which is more targeted |
| A single global bias (not per-head) | Limits expressiveness; different heads could learn different voice-interaction patterns |
| A fixed positive/negative bias | Gives up flexibility; better to let the model learn which relations help |

### 9.3 Conditional track editing

**Motivation.** If both voices are corrupted during training, the model only ever learns
"recover both tracks from noise at once". The commoner scenario in real composition is:
**the melody is given, generate or repair the accompaniment**.

**How the released code does it.** The accompaniment-editing and accompaniment-inpainting
datasets perturb or mask the accompaniment track only, and leave the melody intact as
context. The constraint then falls out of the data construction and the editable-flag
mechanism, with **no change to the loss function or the model architecture**:

1. `editable` / `token_editable` marks which tokens inside the region may be edited (target voice = True, everything else = False).
2. Frozen tokens keep their value throughout intermediate-state sampling — never deleted, never replaced.
3. The Levenshtein DP therefore emits `del = 0, ins = 0` for the frozen tokens automatically, since they are identical in the intermediate state and in the target.
4. At inference the input is the **complete sequence** (the region is not cut out), the frozen voice stays as context, `editable` is True only for the target voice's tokens, and no `insertion_seed` is needed because the region already contains editable tokens. The ordinary delete → insert → fill loop is reused unchanged.

**The capability gained**: one model, several uses — accompaniment editing, accompaniment
generation under a given melody, and (symmetrically) melody repair under a given
accompaniment.

### 9.4 Voice identification per encoding scheme

| Scheme | Voice marker | Strategy |
|--------|--------------|----------|
| A (abs/separated) | `TRACK0_START` / `TRACK1_START` | Switch state on the marker |
| B (rel/separated) | No explicit marker | Beat alternation: even = melody, odd = accompaniment |
| C (rel/bundled) | `SPLIT_0` / `SPLIT_1` | Switch on the marker, plus EMPTY alternation |
| D (abs/bundled) | `SPLIT_0` / `SPLIT_1` | Same as Scheme C |

The `SCHEME_TOKENS` table in `src/iteredit/data/dataset_editing.py` records the marker IDs
per scheme (`track0` / `track1`; note that Scheme B has none, so its voices are recovered
positionally). Because the beat-level interleaving is positional, separating the
accompaniment is a matter of indexing.

---

## 10. Evaluation Design

### 10.1 Baselines

All methods use **exactly the same deterministic mask positions** (the middle 30% of the
beats, `seed = 42`) so that the comparison is fair.

The baselines reported in the paper are:

| Baseline | Method | Question it answers |
|----------|--------|---------------------|
| **No-Edit** | Return the input unchanged | Is the model doing anything at all? Calibrates the corruption difficulty |
| **Copy-Ctx** | Copy a random context beat into the corrupted region | Does the model beat copy-and-paste? Music is repetitive, so this is not a weak baseline |
| **BERT-CMLM** | Re-predict every position by masked language modelling (parallel filling, linearly decaying remask) | Iterative editing vs. parallel filling: a comparison *within* the non-autoregressive family |
| **AR (LLaMA)** | ~162 M causal LM trained from scratch on the same data; AR-Detect uses SeqTag's detector for beat-level localization | Does editing beat autoregressive regeneration given the same localization? |
| **Anticipatory** | ~128 M AR model with bidirectional infilling (public weights) | A purpose-built music-infilling AR model, on completion |
| **Diffusion (D3PM)** | ~34 M discrete diffusion, SDEdit-style noise-then-denoise | Editing vs. the diffusion paradigm |

A **vanilla IterEdit** configuration (contiguous masking, without multi-level
perturbation) also appears in the paper's ablation, and answers whether the multi-level
perturbation training is necessary. It is.

**The BERT-CMLM decoding algorithm** (Ghazvininejad et al., 2019):
1. Initialize: `context_left + [MASK]×gt_length + context_right`.
2. Each round $t$ (of $T = 10$): BERT forward → argmax prediction + confidence (max softmax probability).
3. Keep the highest-confidence predictions; re-mask the $\lfloor n \times \frac{T-t-1}{T} \rfloor$ lowest-confidence positions.
4. On the final round, take argmax everywhere and do not re-mask.

### 10.2 Metrics

The paper's headline metrics, computed by `evaluation/metrics.py`, are beat exact match,
note F1, mean pitch error, chroma F1, rhythm similarity, context preservation, and FMD
(Fréchet Music Distance).

IterEdit additionally has a token-level evaluator of its own
(`src/iteredit/evaluation/evaluate.py`) used for development and diagnosis:

| Dimension | Metric | Meaning |
|-----------|--------|---------|
| Sequence accuracy | Token Accuracy | Fraction of positions matching exactly |
| Sequence accuracy | Normalized Edit Distance | Edit distance / target length; lower is better |
| Pitch / position | Pitch Accuracy | Compares only the position component of the token (`token // 81`) — for Scheme C this is the *relative* position component |
| Rhythm | Pattern Accuracy | Compares only the ternary-pattern component (`token % 81`) |
| Structure | Length Accuracy | Generated length / target length |
| Structure | Note Density Ratio | Note density of the repaired region / note density of the context |
| Efficiency | Average Iterations | Rounds to convergence (IterEdit only) |

**Separating pitch from rhythm** evaluates the two orthogonal dimensions of a note
independently, which diagnoses whether the model is stronger on one than the other.

### 10.3 Ablations

| Ablation | Comparison | Hypothesis under test | Outcome |
|----------|------------|-----------------------|---------|
| Multi-level perturbation vs. contiguous masking | Full vs. vanilla | Is multi-level perturbation needed to generalize to fine-grained editing? | Yes: .480 vs. .114 on editing (≈4×) |
| Encoder-only vs. encoder–decoder | Enc-Dec cross-attention variant | Is the encoder-only inductive bias load-bearing? | Yes: Enc-Dec collapses (.051 / .034) |
| BERT initialization vs. random | Train from scratch | What is pre-training worth? | Large: scratch loses ~70% of correction beat |
| With / without track-aware bias | §9.2 | Does an explicit voice-relation bias help? | **No** — it degrades both tasks (§9.2) |
| Encoding scheme | A / B / C / D | Does encoding choice matter for editing? | Substantially, with a significant encoding × method interaction |

---

## 11. Limitations and Future Directions

| Limitation | Analysis | Status |
|------------|----------|--------|
| Train–inference mismatch | Training rolls in from random corruption; inference rolls in from the model's own output | Open (DAgger-style roll-in) |
| Fixed corruption distribution | 0.3 / 0.2 / 0.5 may not be optimal | Open (curriculum learning) |
| Soft context constraint in the completion mode | A loss mask is not a hard freeze | Open (hard constraint at training time) |
| `max_insert = 20` may be generous | The empirical distribution is probably concentrated in 0–10 | Open (data-driven upper bound) |
| ~~Whole-beat masking only~~ | ~~Cannot do single-track editing~~ | **Resolved** (§9.3: accompaniment-editing and accompaniment-inpainting datasets) |
| Voice markers are not uniform across schemes | The four schemes identify voices differently (§9.4); Scheme B has no explicit marker | Known; handled by the per-scheme `SCHEME_TOKENS` table |
| Weak on segment completion | Iterative refinement from an empty span is a poor fit; TagFill is the right tool there (paper §5.2) | By design — this delineates the edit-density boundary |

---

## References

1. Gu, J., Wang, C., & Zhao, J. (2019). Levenshtein Transformer. *NeurIPS*.
2. Ghazvininejad, M., Levy, O., Liu, Y., & Zettlemoyer, L. (2019). Mask-Predict: Parallel Decoding of Conditional Masked Language Models. *EMNLP*.
3. Huang, C. A., et al. (2019). Music Transformer: Generating Music with Long-Term Structure. *ICLR*.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
6. Thickstun, J., et al. (2024). Anticipatory Music Transformer. *TMLR*.
7. Austin, J., et al. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). *NeurIPS*.
