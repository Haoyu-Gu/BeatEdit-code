# BeatEdit: Symbolic Music Generation as Explicit Editing

Official implementation for the ACM Multimedia 2026 paper. 

- **Demo page** (audio samples, piano rolls, MIDI player): https://haoyu-gu.github.io/BeatEdit/
- Demo repository: https://github.com/Haoyu-Gu/BeatEdit

## Paper-to-Code Mapping

| Paper Name | Directory | Code Class | Paper Section |
|------------|-----------|------------|---------------|
| **Beat Encoding** | `src/encoding/` | `PianoRollTokenizer` | &sect;2 |
| **Music BERT** | `src/pretraining/` | `BertForMaskedLM` (HuggingFace) | &sect;3.1 |
| **SeqTag** | `src/seqtag/` | `MusicGECToR` | &sect;3.2 |
| **IterEdit** | `src/iteredit/` | `LevenshteinTransformer` | &sect;3.3 |
| **TagFill** | `src/tagfill/` | `FELIXTagger` + `FELIXInserter` | &sect;3.4 |
| **CPWord baseline** | `src/baselines/cpword/` | &mdash; | &sect;4.2 |
| **REMI baseline** | `src/baselines/remi/` | &mdash; | &sect;4.2 |
| **Decode-Filter-Reencode** | `src/seqtag/scheme_*/inference.py` (`post_process`) | &mdash; | Appendix E/F |
| **Perturbation / edit labels** | `src/{seqtag,tagfill}/scheme_*/perturbation.py`, `label_extractor.py` | &mdash; | Appendix B/G |
| **Score → npz preprocessing** | `data_prep/` | `XMLToNPZ` | &sect;4.1 |

### Encoding Schemes (2 x 2 factorial design)

| Scheme | Position | Token Org. | Vocab | Tok/Note | Avg. Seq Len |
|--------|----------|------------|-------|----------|--------------|
| A | Absolute | Separated | 186 | 2 | 1,907 |
| B | Relative | Separated | 185 | 2 | 1,891 |
| C | Relative | Bundled | 7,145 | 1 | 1,163 |
| D | Absolute | Bundled | 7,145 | 1 | 1,163 |

**Encoding conventions.** The ternary pattern digits follow the paper
(Appendix A): `0` = silent, `1` = onset, `2` = sustain continuation, so a
quarter note is pattern `53` = (1,2,2,2) and a pure continuation beat is
`80` = (2,2,2,2). Bundled tokens are `position × 81 + pattern`
(Scheme C: relative position; Scheme D: absolute pitch index).

**Two vocab numbers.** The vocab sizes above (and in the paper) refer to the
model vocabulary in `src/pretraining/scheme_*/config.py` (base vocabulary +
`[MASK]`). The `vocab_size` fields inside `src/encoding/scheme_*/config.py`
are legacy values of the encoding module (e.g. Scheme A lists 268 because of
reserved slots for an old autoregressive generator) &mdash; the effective token
range is the same; see the `NOTE` in those files.

## Directory Structure

```
BeatEdit/
├── README.md  LICENSE  requirements.txt
│
├── src/                         # Source code
│   ├── encoding/                # Beat Encoding (4 schemes)
│   │   └── scheme_{A,B,C,D}/   #   my_tokenizer.py, config.py, token2midi.py, PianoDataset.py
│   ├── pretraining/             # Music BERT MLM pre-training
│   │   └── scheme_{A,B,C,D}/   #   train_mlm.py, config.py, mlm_dataset.py, my_tokenizer.py
│   ├── seqtag/                  # SeqTag (error correction)
│   │   ├── scheme_{A,B,C,D}/   #   model.py, train_gector.py, inference.py, evaluate.py,
│   │   │                        #   perturbation.py, label_extractor.py, sequence_parser.py
│   │   └── remi_variant/       #   REMI encoding comparison
│   ├── iteredit/                # IterEdit (accompaniment editing + completion)
│   │   ├── models/              #   levenshtein_transformer.py
│   │   ├── data/                #   dataset.py, dataset_editing.py, dataset_accomp_inpainting.py,
│   │   │                        #   levenshtein_utils.py, masking.py, sequence_parser.py, tokenizer.py
│   │   ├── training/            #   train.py, train_editing.py, train_accomp_inpainting.py
│   │   ├── inference/           #   pipeline.py
│   │   ├── configs/             #   config.py
│   │   └── evaluation/          #   evaluate.py
│   ├── tagfill/                 # TagFill (segment completion)
│   │   └── scheme_{A,B,C,D}/   #   models/{tagger,inserter}.py, inference/pipeline.py,
│   │                            #   data/, training/, utils/
│   └── baselines/               # External encoding baselines
│       ├── cpword/              #   CPWord comparison
│       └── remi/                #   REMI comparison
│
├── evaluation/                  # Evaluation framework
│   ├── metrics.py               #   beat_exact_match, note_f1, MPE, chroma_f1, FMD
│   ├── evaluate.py              #   Main evaluation entry point
│   ├── scheme_utils.py          #   Unified loader for 4 encoding schemes
│   ├── statistical_tests.py     #   Bootstrap + Wilcoxon significance tests
│   ├── summarize.py             #   Paper table generation
│   ├── anova_and_pairwise.py    #   Two-way ANOVA
│   ├── benchmark_speed.py       #   Inference latency (see "Efficiency" below)
│   ├── reeval_decoded_beat_exact.py    # decoded-note-space beat metric (see "Metrics" below)
│   ├── pairwise_bootstrap_decoded.py   # significance tests in decoded space
│   └── verify_filter_roundtrip.py      # Decode-Filter-Reencode validity check (App. E/F)
│
├── results/                     # Pre-computed experimental results (JSON)
│   ├── correction/  editing/  completion/   # per-task, per-method, per-scheme
│   ├── baselines/               #   LLaMA + Diffusion baseline results
│   ├── significance/            #   Statistical test results
│   ├── cascade_analysis/        #   Cascade error analysis
│   ├── master_statistics.json   #   Aggregated results (155 groups)
│   └── benchmark_results.json   #   Inference speed data
│
├── data_prep/                   # Score -> npz preprocessing
│   ├── xml2npz.py               #   MusicXML (MuseScore) -> npz
│   └── midi2npz.py              #   MIDI (Lakh) -> npz
├── scripts/                     # Step-by-step reproduction scripts (00-07)
├── checkpoints/                 # Model weights (populate after training)
└── docs/                        # ENCODING_SPEC, TRAINING_OVERVIEW, METHOD_LevT, ...
```

The subjective evaluation (paper's listening study) is showcased on the
[demo page](https://haoyu-gu.github.io/BeatEdit/); rating data is not
distributed for participant-privacy reasons.

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
bash scripts/00_setup.sh
```

### 2. Data Preparation
```bash
bash scripts/01_download_data.sh          # download guide (MuseScore, Lakh)

# Convert scores to the npz piano-roll format the training code reads
python data_prep/xml2npz.py  /path/to/musescore/ --output-dir /path/to/data/npz --workers 8
export BEATEDIT_DATA_DIR=/path/to/data/npz
```
`data_prep/` also has `midi2npz.py` for MIDI sources, and a `--velocity` flag
that emits 6-channel arrays (velocity is off by default, as in the paper).
See [data_prep/README.md](data_prep/README.md) for the format and its caveats.

### 3. Training Pipeline

Training follows a strict dependency order:

```
BERT Pre-training (§3.1)
    ├── SeqTag (§3.2)      — Stage I (frozen) → Stage III (clean mixing)
    ├── IterEdit (§3.3)    — inpainting mode + editing mode
    └── TagFill (§3.4)     — tagger (Focal Loss) → inserter (BERT-init MLM)
```

```bash
# Step 1: Pre-train BERT (all 4 schemes)
for s in A B C D; do
    SCHEME=$s DATA_DIR=/path/to/data/npz bash scripts/02_pretrain_bert.sh
done

# Step 2: Train methods (can run in parallel across schemes).
# BERT_CKPT is the checkpoint *directory* written by step 1 — it holds
# model.safetensors. Use best_model/ once evaluation has run, final_model/
# otherwise.
BERT_A=checkpoints/bert/scheme_A/best_model
BERT_D=checkpoints/bert/scheme_D/best_model

SCHEME=A DATA_DIR=/path/to/data/npz BERT_CKPT=$BERT_A \
    bash scripts/03_train_seqtag.sh      # Stage I -> Stage III

SCHEME=A DATA_DIR=/path/to/data/npz BERT_CKPT=$BERT_A \
    bash scripts/05_train_tagfill.sh     # tagger -> inserter

DATA_DIR=/path/to/data/npz BERT_CKPT=$BERT_D \
    bash scripts/04_train_iteredit.sh    # inpainting -> editing
```

**Pilot run.** The whole pipeline runs on a laptop (CPU) with a shrunken
backbone — useful to check the setup before committing GPU hours:

```bash
BEATEDIT_LAYERS=2 BEATEDIT_HIDDEN=128 BEATEDIT_HEADS=4 BEATEDIT_FFN=256 \
BEATEDIT_EPOCHS=2 BEATEDIT_BATCH=4 \
    SCHEME=A DATA_DIR=/path/to/data/npz bash scripts/02_pretrain_bert.sh
```
The same `BEATEDIT_*` values must be set for every downstream stage, since the
model has to match the checkpoint it is initialized from. fp16 is downgraded
automatically when CUDA is unavailable (`BEATEDIT_PRECISION` overrides).

### 4. Evaluation
```bash
bash scripts/06_evaluate_all.sh     # Run metrics + significance tests
bash scripts/07_generate_tables.sh  # Generate paper tables
```

### 5. Encoding validity check (no training required)
```bash
python evaluation/verify_filter_roundtrip.py --n 200 --scheme B
python evaluation/verify_filter_roundtrip.py --n 200 --scheme A
```
Verifies the Decode-Filter-Reencode post-processing (paper Appendix E/F)
over 200 random edit sequences including cuts into long notes:
out-of-region beats preserved verbatim (100%), zero violations,
100% decodable.

## Developer conveniences

```bash
make setup      # create .venv, install deps, run smoke tests
make demo       # side-by-side encoding of the same piece under schemes A-D
make verify     # syntax check + encoding tests + Appendix E/F validity check
make pipeline SCHEME=A DATA_DIR=/path/to/data   # full pipeline for one scheme
```

`tools/encoding_demo.py` encodes any note spec under all four schemes and
prints the token sequences next to each other &mdash; the quickest way to see how
the 2×2 design (absolute/relative × separated/bundled) plays out, and to
sanity-check round-trips. `tests/test_encoding.py` runs numpy-only encoding
tests (also wired into CI).

**Model-size knobs.** The backbone can be resized without editing any config
file &mdash; useful for pilot runs on small GPUs:

```bash
BEATEDIT_LAYERS=4 BEATEDIT_HIDDEN=256 BEATEDIT_EPOCHS=3 \
    SCHEME=A DATA_DIR=/path/to/data bash scripts/02_pretrain_bert.sh
```

Recognized overrides: `BEATEDIT_LAYERS`, `BEATEDIT_HIDDEN`, `BEATEDIT_HEADS`,
`BEATEDIT_FFN`, `BEATEDIT_EPOCHS`, `BEATEDIT_BATCH`, `BEATEDIT_DATA_DIR`.

## Metrics: token space vs. decoded space

`evaluation/metrics.py` computes beat exact match in **token space**.
For cross-scheme comparability, the paper's editing-task beat numbers use the
**decoded-note-space** re-evaluation (`evaluation/reeval_decoded_beat_exact.py`),
which decodes each beat to sorted `(pitch, pattern)` notes before comparison;
correction-task numbers are identical in both spaces. Re-running the decoded
re-evaluation requires per-sample prediction dumps
(`evaluation/predictions/<task>/<method>/<scheme>/*.json`), which are
regenerated by running inference (scripts 03-06) and are not shipped with the
repository; the resulting aggregates are pre-computed in `results/`.

## Efficiency benchmarks

`evaluation/benchmark_speed.py` reproduces SeqTag and TagFill latency
measurements out of the box (given checkpoints). The LevT, LLaMA, CMLM and
diffusion baseline benchmarks depend on baseline packages that are not part
of this release and are skipped with a notice; their timings are included in
`results/benchmark_results.json`.

## Model Architecture Summary

All methods share the same BERT backbone (~26.6M parameters):

| Component | Config |
|-----------|--------|
| Layers / Heads / Hidden / FFN | 8 / 8 / 512 / 2,048 |
| Parameters | ~26.6M |
| Vocab size | 185 (Scheme B) to 7,145 (Scheme C/D) |
| Max position embeddings | 2,048 |

The Levenshtein Transformer (IterEdit) is a self-contained re-implementation
on top of the HuggingFace BERT backbone; the edit-label alignment algorithms
in `src/iteredit/data/levenshtein_utils.py` are ported from
[fairseq](https://github.com/facebookresearch/fairseq)'s
`levenshtein_utils.py` to pure numpy. fairseq is **not** a dependency.

### Training Hyperparameters (Paper Appendix C/D)

| Component | LR | Batch (eff.) | Epochs | Optimizer | Schedule |
|-----------|----|-------------|--------|-----------|----------|
| BERT Pre-training | 1e-4 | 256 | 30 | AdamW (wd=0.01) | Cosine + 10% warmup |
| SeqTag Stage I | 1e-5 (BERT) / 1e-4 (head) | 64 | 20 | AdamW | Cosine + 10% warmup |
| SeqTag Stage III | 5e-6 | 64 | 3 | AdamW | Cosine + 10% warmup |
| IterEdit | 3e-4 | 64 | 30 | AdamW (wd=0.01) | Cosine + 10% warmup |
| TagFill Tagger | 1e-4 | 96 (32×3) | 30 | AdamW (wd=0.01) | Cosine + 10% warmup |
| TagFill Inserter | 1e-4 | 96 (32×3) | 30 | AdamW (wd=0.01) | Cosine + 10% warmup |

## Pre-computed Results

All experimental results are preserved as JSON files in `results/`. The
aggregated `master_statistics.json` contains 155 groups of metrics. LevT
result files carry the internal run tag `levt_editing_v2` &mdash; these are the
runs reported in the paper.

```python
import json
with open('results/master_statistics.json') as f:
    stats = json.load(f)
print(f"Total result groups: {len(stats['results'])}")
```

## Hardware Requirements

- **Training**: 2x GPU with 24GB VRAM each (paper used this setup)
- **Inference**: Single GPU (37-655 ms per sample)
- **Evaluation of pre-computed results**: CPU only

Training time estimates (2x GPU):
- BERT Pre-training: 4.6h (Scheme D) to 27.3h (Scheme B)
- SeqTag: ~6-8h per scheme (Stage I) + ~1h (Stage III)
- IterEdit: ~10-15h
- TagFill: ~8-10h per scheme (tagger) + ~8-10h (inserter)

## Authors

Haoyu Gu<sup>1</sup>, Lekai Qian<sup>1</sup>, Haowu Zhou<sup>1</sup>,
Qi Liu<sup>1,\*</sup>, Shuai Wang<sup>2,\*</sup>

<sup>1</sup> School of Future Technology, South China University of Technology, Guangzhou, China
<sup>2</sup> School of Intelligence Science and Technology, Nanjing University, Suzhou, China

<sup>\*</sup> Corresponding authors.

## Citation

```bibtex
@inproceedings{gu2026beatedit,
  title     = {BeatEdit: Symbolic Music Generation as Explicit Editing},
  author    = {Gu, Haoyu and Qian, Lekai and Zhou, Haowu and Liu, Qi and Wang, Shuai},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)).
