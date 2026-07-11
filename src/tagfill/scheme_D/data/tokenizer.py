"""
Tokenizer wrapper for FELIX-Music (Scheme D: absolute_bundled).

Imports PianoRollTokenizer from the Scheme D encoding module via importlib.
"""

import os
import sys
import importlib.util

# Import PianoRollTokenizer from src/encoding/scheme_D
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MUSIC_BERT_DIR = os.path.join(_SRC_DIR, 'encoding', 'scheme_D')
_spec = importlib.util.spec_from_file_location(
    "encoding_scheme_D_tokenizer",
    os.path.join(_MUSIC_BERT_DIR, "my_tokenizer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
# Register the module so DataLoader workers (spawn start method) can
# unpickle classes defined in it.
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
PianoRollTokenizer = _mod.PianoRollTokenizer


def create_tokenizer():
    """Create a PianoRollTokenizer with standard absolute_bundled config."""
    return PianoRollTokenizer(
        patch_h=1,
        patch_w=4,
        pattern_num=81,
        beats_length=88,
    )
