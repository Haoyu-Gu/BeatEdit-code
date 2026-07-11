"""
Tokenizer wrapper for FELIX-Music (Scheme A: no_pair).

Imports PianoRollTokenizer from the Scheme A encoding module via importlib.
"""

import os
import importlib.util

# Import PianoRollTokenizer from src/encoding/scheme_A
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MUSIC_BERT_DIR = os.path.join(_SRC_DIR, 'encoding', 'scheme_A')
_spec = importlib.util.spec_from_file_location(
    "encoding_scheme_A_tokenizer",
    os.path.join(_MUSIC_BERT_DIR, "my_tokenizer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PianoRollTokenizer = _mod.PianoRollTokenizer


def create_tokenizer():
    """Create a PianoRollTokenizer with standard no_pair config."""
    return PianoRollTokenizer(
        patch_h=1,
        patch_w=4,
        pattern_num=81,
        beats_length=88,
    )
