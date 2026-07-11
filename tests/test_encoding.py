#!/usr/bin/env python3
"""Numpy-only encoding tests (no torch required).

Covers, for all four schemes:
  - paper example pattern values (Appendix A.2)
  - encode -> compress -> decompress -> decode round-trip
  - bundled token composition (position * 81 + pattern)

Run directly (`python tests/test_encoding.py`) or via pytest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from encoding_demo import SCHEMES, beats_to_pianoroll, load_tokenizer  # noqa: E402


def test_paper_pattern_values():
    """Quarter=53, two eighths=50, staccato 16th=27, continuation=80."""
    tok = load_tokenizer('A')
    for pattern in (53, 50, 27, 80, 5):
        roll = beats_to_pianoroll([[[39, pattern]]])
        matrix = tok.image_to_patch_tokens(roll, strict_mode=False)
        assert int(matrix[0, 39]) == pattern, (pattern, int(matrix[0, 39]))


def test_roundtrip_all_schemes():
    rng = np.random.RandomState(0)
    for scheme, info in SCHEMES.items():
        tok = load_tokenizer(scheme)
        for _ in range(25):
            beats = []
            for _ in range(rng.randint(2, 6)):
                pitches = rng.choice(88, rng.randint(1, 4), replace=False)
                beats.append([[int(p), int(rng.choice([53, 50, 27, 80]))]
                              for p in sorted(pitches)])
            roll = beats_to_pianoroll(beats)
            matrix = tok.image_to_patch_tokens(roll, strict_mode=False)
            seq = tok.compress_tokens(matrix, **info['markers'])
            restored = tok.decompress_tokens(seq, **info['markers'])
            assert np.array_equal(tok.patch_tokens_to_image(restored), roll), scheme


def test_bundled_composition():
    """Scheme D bundles absolute_pitch * 81 + pattern into one token."""
    tok = load_tokenizer('D')
    roll = beats_to_pianoroll([[[39, 53]]])
    matrix = tok.image_to_patch_tokens(roll, strict_mode=False)
    seq = tok.compress_tokens(matrix, **SCHEMES['D']['markers'])
    assert list(seq) == [7129, 39 * 81 + 53], list(seq)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"{name}: PASS")
    print("All encoding tests passed.")
