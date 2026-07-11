# Piano Music Token Encoding Specification

This document describes the four encoding schemes (2×2 ablation) used in this project for representing piano music as token sequences. All four schemes share the same input data format, patch tokenization method, and overall sequence structure, but differ in how they compress beat-level note information into flat token sequences.

---

## 1. Shared Components (All Three Schemes)

### 1.1 Input Data

Each music file is an `.npz` containing multiple measures. Each measure is a numpy array of shape `(4, 88, t)`:
- **Channels 0-1**: Track 0 (high voice / right hand) — channel 0 = sustain, channel 1 = onset
- **Channels 2-3**: Track 1 (low voice / left hand) — channel 0 = sustain, channel 1 = onset
- **88**: Piano keys (pitch axis, high-to-low after reversal)
- **t**: Time steps within the measure

Each cell is binary (0 or 1): sustain=1 means the key is held, onset=1 means a new note strike at that time step. Onset can only be 1 where sustain is also 1.

### 1.2 Patch Tokenization (Ternary Encoding)

Before compression, each beat of each track is converted to **patch tokens** using a ternary (base-3) encoding:

- **patch_h = 1, patch_w = 4**: Each patch covers 1 pitch position x 4 time steps
- For each patch, the two channels (sustain + onset) are combined into a ternary digit:
  - `0`: no note (sustain=0, onset=0)
  - `1`: sustain only (sustain=1, onset=0)
  - `2`: onset + sustain (sustain=1, onset=1)
- Four ternary digits are encoded as a single base-3 number: `value = d0*27 + d1*9 + d2*3 + d3`
- This produces **81 possible patch token values** (0 to 80), representing `3^4 = 81` patterns

After patch tokenization, each beat (which has `patch_w=4` time steps) of one track becomes a 1D array of 88 patch token values (one per pitch position). Most values are 0 (silence).

### 1.3 Overall Sequence Structure

All three schemes produce a flat token sequence with the same high-level structure:

```
[BOS] [TIME_SIG] [BPM] [BAR] [beat_content...] [BAR] [beat_content...] ... [EOS]
```

- **BOS**: Beginning of sequence token (only for the first segment of a piece)
- **TIME_SIG**: Time signature token (e.g., 4/4, 3/4, 2/4, 6/8, 2/2)
- **BPM**: Tempo token (slow <90, medium 90-200, fast >200, or unknown)
- **BAR**: Bar separator token, placed before each measure's content
- **beat_content**: Within each bar, beats are interleaved at the beat level between the two tracks:
  ```
  [Track0_Beat0] [Track1_Beat0] [Track0_Beat1] [Track1_Beat1] ...
  ```
- **EOS**: End of sequence token (only if the piece is complete, not a continuation)

The three schemes differ ONLY in how each `[TrackX_BeatY]` segment is encoded.

### 1.4 Data Augmentation

- **Pitch shift**: 70% probability of random shift between -5 and +5 semitones
- **Truncation**: Sequences exceeding 2048 tokens are randomly cropped (from start, end, or middle), with the first 8% of non-start crops having their labels masked

---

## 2. Scheme A: `no_pair` (Absolute Position, Separated)

**Directory**: `no_pair/`
**Vocab size**: 186
**Tokens per note**: 2 (position + value)

### 2.1 Beat Encoding

Each beat of each track is compressed using **absolute position encoding with a track marker prefix**:

- **Non-empty beat**: `[TRACK_MARKER] [abs_pos_0] [val_0] [abs_pos_1] [val_1] ...`
  - `TRACK_MARKER`: Track 0 uses `TRACK0_START (183)`, Track 1 uses `TRACK1_START (184)`
  - `abs_pos_i = 81 + pitch_index` (range: 81 to 168, since pitch 0-87)
  - `val_i`: The patch token value at that pitch position (0 to 80)
  - Only non-zero positions are encoded (zero-suppression)
  - Positions are listed in ascending order of pitch index

- **Empty beat** (all zeros): `[TRACK_MARKER] [0]`
  - The marker is followed by a literal `0` to indicate no notes

### 2.2 Example

A beat with notes at pitch positions 3 (value=66) and 10 (value=40) for Track 0:
```
[183] [84] [66] [91] [40]
```
Where: 183=TRACK0_START, 84=81+3, 91=81+10

### 2.3 Token ID Allocation

| Range | Meaning |
|-------|---------|
| 0-80 | Patch token values (ternary patterns) |
| 81-168 | Absolute position markers (81 + pitch_index) |
| 170 | BAR token |
| 171 | EOS token |
| 172 | BOS token |
| 173 | PAD token |
| 174-178 | Time signature tokens (5 types) |
| 179-182 | BPM tokens (4 categories) |
| 183 | TRACK0_START marker |
| 184 | TRACK1_START marker |

### 2.4 Properties

- Track identity is explicit via distinct start markers (183 vs 184)
- Each note costs exactly 2 tokens (position + value)
- Empty beats cost 2 tokens (marker + 0)
- Position encoding is absolute (each position is independent)

---

## 3. Scheme B: `no_pair_related` (Relative Position, Separated)

**Directory**: `no_pair_related/`
**Vocab size**: 185
**Tokens per note**: 2 (relative position + value)

### 3.1 Beat Encoding

Each beat uses **relative position encoding** with explicit empty/end markers instead of track markers:

- **Non-empty beat**: `[81+rel_pos_0] [val_0] [81+rel_pos_1] [val_1] ... [END_MARKER]`
  - The first position is relative to 0 (effectively absolute)
  - Each subsequent position is relative to the previous non-zero position
  - `rel_pos_i = current_pitch_index - previous_pitch_index`
  - The beat ends with `END_MARKER (170)` to signal completion
  - No track-specific prefix; track identity is inferred from position in the interleaving pattern (even index = Track 0, odd index = Track 1)

- **Empty beat**: `[EMPTY_MARKER]`
  - A single token `EMPTY_MARKER (169)` represents a completely silent beat

### 3.2 Example

A beat with notes at pitch positions 3 (value=66), 10 (value=40), and 78 (value=7):
```
[84] [66] [88] [40] [149] [7] [170]
```
Where:
- 84 = 81+3 (first note, relative to 0, so rel_pos=3)
- 66 = patch value at pitch 3
- 88 = 81+7 (second note, relative to pitch 3, so rel_pos=10-3=7)
- 40 = patch value at pitch 10
- 149 = 81+68 (third note, relative to pitch 10, so rel_pos=78-10=68)
- 7 = patch value at pitch 78
- 170 = END_MARKER

### 3.3 Token ID Allocation

| Range | Meaning |
|-------|---------|
| 0-80 | Patch token values (ternary patterns) |
| 81-168 | Relative position markers (81 + relative_distance, max distance=87) |
| 169 | EMPTY_MARKER (empty beat) |
| 170 | END_MARKER (end of non-empty beat) |
| 171 | BAR token |
| 172 | EOS token |
| 173 | BOS token |
| 174 | PAD token |
| 175-179 | Time signature tokens (5 types) |
| 180-183 | BPM tokens (4 categories) |

### 3.4 Properties

- No explicit track markers; track identity is determined by interleaving order
- Each note costs 2 tokens (relative position + value), same as `no_pair`
- Non-empty beats additionally cost 1 token for END_MARKER
- Empty beats cost only 1 token (EMPTY_MARKER), saving 1 token vs `no_pair`
- Relative positions may help the model learn local pitch intervals
- Maximum relative distance = 87 (full keyboard span), fitting in range 81-168

---

## 4. Scheme C: `with_pair` (Bundled Encoding)

**Directory**: `with_pair/`
**Vocab size**: 7145
**Tokens per note**: 1 (bundled position+value)

### 4.1 Beat Encoding

Each beat uses **bundled encoding** that packs the relative position and patch value into a single token:

- **Non-empty beat**: `[SPLIT_X] [bundled_0] [bundled_1] ...`
  - `SPLIT_X`: Track-specific split marker — Track 0 uses `SPLIT_0 (7129)`, Track 1 uses `SPLIT_1 (7130)`
  - `bundled_i = relative_position * 81 + patch_token_value`
  - Relative position works the same as in `no_pair_related` (first note relative to 0, subsequent notes relative to previous position)
  - Bundled token range: 0 to 7127 (max relative_pos=87, max value=80: 87*81+80=7127)
  - No end marker needed; a beat ends when the next special token (>= 7128) is encountered

- **Empty beat**: `[EMPTY_MARKER]`
  - A single token `EMPTY_MARKER (7128)` represents a completely silent beat
  - Note: empty beats use the same token regardless of which track they belong to

### 4.2 Example

A beat with notes at pitch positions 3 (value=66) and 10 (value=40) for Track 0:
```
[7129] [309] [607]
```
Where:
- 7129 = SPLIT_0 (Track 0 marker)
- 309 = 3*81 + 66 (first note: rel_pos=3, value=66)
- 607 = 7*81 + 40 (second note: rel_pos=7 from pitch 3 to pitch 10, value=40)

### 4.3 Token ID Allocation

| Range | Meaning |
|-------|---------|
| 0-7127 | Bundled tokens (relative_pos * 81 + patch_value) |
| 7128 | EMPTY_MARKER (empty beat) |
| 7129 | SPLIT_0 (Track 0 non-empty beat prefix) |
| 7130 | SPLIT_1 (Track 1 non-empty beat prefix) |
| 7131 | BAR token |
| 7132 | EOS token |
| 7133 | BOS token |
| 7134 | PAD token |
| 7135-7139 | Time signature tokens (5 types) |
| 7140-7143 | BPM tokens (4 categories) |

### 4.4 Properties

- Most compact: each note costs only 1 token (bundled)
- Explicit track markers (SPLIT_0 vs SPLIT_1) for non-empty beats
- No end marker needed (boundary detected by special token IDs >= 7128)
- Trade-off: much larger vocabulary (7144 vs 184/268), leading to higher initial cross-entropy loss and larger embedding matrix
- The bundled token encodes both "where" and "what" in a single integer

---

## 5. Scheme D: `absolute_bundled` (Absolute Position, Bundled)

**Directory**: `absolute_bundled/`
**Vocab size**: 7145
**Tokens per note**: 1 (bundled position+value)

### 5.1 Beat Encoding

Each beat uses **bundled encoding** that packs the absolute position and patch value into a single token:

- **Non-empty beat**: `[SPLIT_X] [bundled_0] [bundled_1] ...`
  - `SPLIT_X`: Track-specific split marker — Track 0 uses `SPLIT_0 (7129)`, Track 1 uses `SPLIT_1 (7130)`
  - `bundled_i = absolute_position * 81 + patch_token_value`
  - Absolute position is the pitch index directly (0-87), independent of other notes
  - Bundled token range: 0 to 7127 (max abs_pos=87, max value=80: 87*81+80=7127)
  - No end marker needed; a beat ends when the next special token (>= 7128) is encountered

- **Empty beat**: `[EMPTY_MARKER]`
  - A single token `EMPTY_MARKER (7128)` represents a completely silent beat

### 5.2 Example

A beat with notes at pitch positions 3 (value=66) and 10 (value=40) for Track 0:
```
[7129] [309] [850]
```
Where:
- 7129 = SPLIT_0 (Track 0 marker)
- 309 = 3*81 + 66 (pitch 3, value 66)
- 850 = 10*81 + 40 (pitch 10, value 40)

### 5.3 Token ID Allocation

| Range | Meaning |
|-------|---------|
| 0-7127 | Bundled tokens (absolute_pos * 81 + patch_value) |
| 7128 | EMPTY_MARKER (empty beat) |
| 7129 | SPLIT_0 (Track 0 non-empty beat prefix) |
| 7130 | SPLIT_1 (Track 1 non-empty beat prefix) |
| 7131 | BAR token |
| 7132 | EOS token |
| 7133 | BOS token |
| 7134 | PAD token |
| 7135-7139 | Time signature tokens (5 types) |
| 7140-7143 | BPM tokens (4 categories) |

### 5.4 Properties

- Same compact 1-token-per-note format as Scheme C
- Absolute positions: each note's position is independent, no cascade risk from errors
- Explicit track markers (SPLIT_0 vs SPLIT_1) for non-empty beats
- Same large vocabulary as C (7145)
- Key difference from C: position errors do NOT propagate to subsequent tokens

### 5.5 Comparison with Scheme C

| Aspect | C (relative+bundled) | D (absolute+bundled) |
|--------|---------------------|---------------------|
| Position encoding | Relative (delta from previous) | Absolute (direct pitch index) |
| Error propagation | One position error cascades | Errors are isolated |
| Sequence compressibility | Better (small deltas common) | Worse (full pitch indices) |
| MLM pretraining difficulty | Easier (loss=0.449) | Harder (loss=0.502) |
| Downstream editing | GECToR: C wins (+0.030 F1) | FELIX: D wins (+0.023 beat_match) |

---

## 6. Comparison Summary

| Feature | A `no_pair` | B `no_pair_related` | C `with_pair` | D `absolute_bundled` |
|---------|-----------|-------------------|-------------|---------------------|
| Vocab size | 186 | 185 | 7145 | 7145 |
| Position encoding | Absolute | Relative | Relative (bundled) | Absolute (bundled) |
| Tokens per note | 2 | 2 | 1 | 1 |
| Track identification | Explicit marker (183/184) | By interleave order | Explicit marker (7129/7130) | Explicit marker (7129/7130) |
| Empty beat tokens | 2 (marker + 0) | 1 (EMPTY_MARKER) | 1 (EMPTY_MARKER) | 1 (EMPTY_MARKER) |
| Non-empty beat overhead | 1 (track marker) | 1 (END_MARKER) | 1 (SPLIT marker) | 1 (SPLIT marker) |
| Beat boundary detection | Track marker prefix | EMPTY/END markers | Special token threshold (>=7128) | Special token threshold (>=7128) |
| Sequence length | Medium | Medium | Short | Short |
| Error propagation | None | Position cascade | Position cascade | None |

---

## 7. Complete Sequence Example

For a 4/4 piece at 120 BPM with 2 measures, each measure having 4 beats:

```
[BOS] [TIME_SIG_4/4] [BPM_MEDIUM]
[BAR] [T0_B0] [T1_B0] [T0_B1] [T1_B1] [T0_B2] [T1_B2] [T0_B3] [T1_B3]
[BAR] [T0_B0] [T1_B0] [T0_B1] [T1_B1] [T0_B2] [T1_B2] [T0_B3] [T1_B3]
[EOS]
```

Where `[TX_BY]` is the beat content for Track X, Beat Y, encoded differently by each scheme.

For example, if Track 0, Beat 0 has notes at pitch 5 (value=42) and pitch 20 (value=7), and Track 1, Beat 0 is empty:

**no_pair:**
```
... [BAR] [183, 86, 42, 101, 7] [184, 0] ...
```
(Track0: marker=183, pos=81+5=86, val=42, pos=81+20=101, val=7; Track1: marker=184, empty=0)

**no_pair_related:**
```
... [BAR] [86, 42, 96, 7, 170] [169] ...
```
(Track0: rel_pos=81+5=86, val=42, rel_pos=81+15=96, val=7, end=170; Track1: empty=169)

**with_pair:**
```
... [BAR] [7129, 447, 1222] [7128] ...
```
(Track0: split_0=7129, bundled=5*81+42=447, bundled=15*81+7=1222; Track1: empty=7128)

**absolute_bundled:**
```
... [BAR] [7129, 447, 1627] [7128] ...
```
(Track0: split_0=7129, bundled=5*81+42=447, bundled=20*81+7=1627; Track1: empty=7128)

Note: `with_pair` uses relative position (second note: 15*81+7=1222, delta from pitch 5), while `absolute_bundled` uses absolute position (second note: 20*81+7=1627, direct pitch index).
