# BeatEdit developer entry points.
#
#   make setup                      create .venv and install dependencies
#   make verify                     fast correctness checks (no GPU, no data)
#   make demo                       side-by-side encoding demo (schemes A-D)
#   make pretrain SCHEME=A          pre-train Music BERT for one scheme
#   make seqtag|tagfill SCHEME=A    train a method for one scheme
#   make iteredit                   train IterEdit
#   make eval / make tables         evaluation + paper tables
#   make pipeline SCHEME=A          full pipeline for one scheme (02-07)
#
# Common knobs (env vars, all optional):
#   DATA_DIR / BEATEDIT_DATA_DIR    preprocessed npz data directory
#   BERT_CKPT                       pretrained BERT checkpoint for method training
#   BEATEDIT_LAYERS / BEATEDIT_HIDDEN / BEATEDIT_HEADS / BEATEDIT_FFN
#                                   shrink or grow the backbone without editing configs
#   BEATEDIT_EPOCHS / BEATEDIT_BATCH

SCHEME ?= A
PY     ?= python3

.PHONY: setup verify demo compile-check pretrain seqtag tagfill iteredit eval tables pipeline clean

setup:
	bash scripts/setup_env.sh

verify: compile-check
	$(PY) tools/encoding_demo.py
	$(PY) tests/test_encoding.py
	$(PY) evaluation/verify_filter_roundtrip.py --n 200 --scheme A
	$(PY) evaluation/verify_filter_roundtrip.py --n 200 --scheme B

demo:
	$(PY) tools/encoding_demo.py

compile-check:
	$(PY) -m compileall -q src evaluation tools tests

pretrain:
	SCHEME=$(SCHEME) bash scripts/02_pretrain_bert.sh

seqtag:
	SCHEME=$(SCHEME) bash scripts/03_train_seqtag.sh

iteredit:
	bash scripts/04_train_iteredit.sh

tagfill:
	SCHEME=$(SCHEME) bash scripts/05_train_tagfill.sh

eval:
	bash scripts/06_evaluate_all.sh

tables:
	bash scripts/07_generate_tables.sh

pipeline:
	SCHEME=$(SCHEME) bash scripts/08_full_pipeline.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
