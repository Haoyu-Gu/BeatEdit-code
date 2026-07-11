"""
Scheme Utilities: dynamic loading for all 4 encoding schemes.

Provides unified access to scheme-specific modules (tokenizer, parser, perturbation,
model loading, inference) for both GECToR (SeqTag) and FELIX (TagFill) subsystems.
"""

import os
import sys
import copy
import importlib
import importlib.util
import numpy as np

# ==================== Path Configuration ====================

UNIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(UNIFIED_DIR)  # BeatEdit release-tree root
DATA_DIR = os.environ.get("BEATEDIT_DATA_DIR", "/path/to/data/npz")

# ==================== Scheme Configuration ====================
# Directory / checkpoint layout follows the release tree:
#   src/{pretraining,seqtag,tagfill}/scheme_X/  and  checkpoints/{bert,seqtag}/scheme_X/

SCHEME_INFO = {
    'A': {
        'name': 'Scheme A (absolute, separate)',
        'felix_dir': os.path.join(ROOT_DIR, 'src', 'tagfill', 'scheme_A'),
        'gector_dir': os.path.join(ROOT_DIR, 'src', 'seqtag', 'scheme_A'),
        'bert_dir': os.path.join(ROOT_DIR, 'src', 'pretraining', 'scheme_A'),
        'bert_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'bert', 'scheme_A', 'best_model', 'model.safetensors'),
        'gector_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'seqtag', 'scheme_A', 'best_model'),
        'vocab_size': 186, 'pad_token_id': 173, 'mask_token_id': 185,
        'note_min': 0, 'note_max': 168, 'has_marker': True,
        'gector_max_iterations': 3,
    },
    'B': {
        'name': 'Scheme B (relative, separate)',
        'felix_dir': os.path.join(ROOT_DIR, 'src', 'tagfill', 'scheme_B'),
        'gector_dir': os.path.join(ROOT_DIR, 'src', 'seqtag', 'scheme_B'),
        'bert_dir': os.path.join(ROOT_DIR, 'src', 'pretraining', 'scheme_B'),
        'bert_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'bert', 'scheme_B', 'best_model', 'model.safetensors'),
        'gector_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'seqtag', 'scheme_B', 'best_model'),
        'vocab_size': 185, 'pad_token_id': 174, 'mask_token_id': 184,
        'note_min': 0, 'note_max': 168, 'has_marker': False,
        'gector_max_iterations': 3,
    },
    'C': {
        'name': 'Scheme C (relative, bundled)',
        'felix_dir': os.path.join(ROOT_DIR, 'src', 'tagfill', 'scheme_C'),
        'gector_dir': os.path.join(ROOT_DIR, 'src', 'seqtag', 'scheme_C'),
        'bert_dir': os.path.join(ROOT_DIR, 'src', 'pretraining', 'scheme_C'),
        'bert_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'bert', 'scheme_C', 'best_model', 'model.safetensors'),
        'gector_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'seqtag', 'scheme_C', 'best_model'),
        'vocab_size': 7145, 'pad_token_id': 7134, 'mask_token_id': 7144,
        'note_min': 0, 'note_max': 7127, 'has_marker': True,
        'gector_max_iterations': 2,
    },
    'D': {
        'name': 'Scheme D (absolute, bundled)',
        'felix_dir': os.path.join(ROOT_DIR, 'src', 'tagfill', 'scheme_D'),
        'gector_dir': os.path.join(ROOT_DIR, 'src', 'seqtag', 'scheme_D'),
        'bert_dir': os.path.join(ROOT_DIR, 'src', 'pretraining', 'scheme_D'),
        'bert_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'bert', 'scheme_D', 'best_model', 'model.safetensors'),
        'gector_checkpoint': os.path.join(ROOT_DIR, 'checkpoints', 'seqtag', 'scheme_D', 'best_model'),
        'vocab_size': 7145, 'pad_token_id': 7134, 'mask_token_id': 7144,
        'note_min': 0, 'note_max': 7127, 'has_marker': True,
        'gector_max_iterations': 2,
    },
}


# ==================== Module Loading ====================

_FELIX_PKG_PREFIXES = ('configs', 'data', 'models', 'inference', 'utils')


def _save_felix_modules():
    """Save and remove FELIX-related modules to avoid namespace conflicts."""
    saved = {}
    for k in list(sys.modules.keys()):
        if any(k == p or k.startswith(p + '.') for p in _FELIX_PKG_PREFIXES):
            saved[k] = sys.modules.pop(k)
    return saved


def _restore_felix_modules(saved):
    """Restore previously saved FELIX modules."""
    for k in list(sys.modules.keys()):
        if any(k == p or k.startswith(p + '.') for p in _FELIX_PKG_PREFIXES):
            del sys.modules[k]
    sys.modules.update(saved)
    importlib.invalidate_caches()


def _load_module(module_name, file_path, injected_modules=None):
    """Load a Python module from file path using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)

    mod_dir = os.path.dirname(file_path)
    old_path = sys.path[:]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    saved_modules = {}
    if injected_modules:
        for bare_name, injected_mod in injected_modules.items():
            saved_modules[bare_name] = sys.modules.get(bare_name)
            sys.modules[bare_name] = injected_mod

    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path = old_path
        for bare_name in saved_modules:
            if saved_modules[bare_name] is None:
                sys.modules.pop(bare_name, None)
            else:
                sys.modules[bare_name] = saved_modules[bare_name]

    return mod


# ==================== GECToR Loader ====================

class GECToRLoader:
    """Loads GECToR modules for a single scheme (from scheme_loader.py)."""

    def __init__(self, scheme):
        self.scheme = scheme
        info = SCHEME_INFO[scheme]
        self.gector_dir = info['gector_dir']
        self.checkpoint_path = info['gector_checkpoint']
        self.max_iterations = info['gector_max_iterations']

        prefix = f"gector_{scheme.lower()}"
        inj = {}

        self.config = _load_module(
            f"{prefix}_config",
            os.path.join(self.gector_dir, "config.py"),
        )
        inj['config'] = self.config

        self.sequence_parser = _load_module(
            f"{prefix}_sequence_parser",
            os.path.join(self.gector_dir, "sequence_parser.py"),
            injected_modules=inj,
        )
        inj['sequence_parser'] = self.sequence_parser

        self.perturbation = _load_module(
            f"{prefix}_perturbation",
            os.path.join(self.gector_dir, "perturbation.py"),
            injected_modules=inj,
        )
        inj['perturbation'] = self.perturbation

        self.label_extractor = _load_module(
            f"{prefix}_label_extractor",
            os.path.join(self.gector_dir, "label_extractor.py"),
            injected_modules=inj,
        )
        inj['label_extractor'] = self.label_extractor

        self.model_mod = _load_module(
            f"{prefix}_model",
            os.path.join(self.gector_dir, "model.py"),
            injected_modules=inj,
        )
        inj['model'] = self.model_mod

        self.inference_mod = _load_module(
            f"{prefix}_inference",
            os.path.join(self.gector_dir, "inference.py"),
            injected_modules=inj,
        )
        inj['inference'] = self.inference_mod

        self.dataset_mod = _load_module(
            f"{prefix}_dataset",
            os.path.join(self.gector_dir, "dataset.py"),
            injected_modules=inj,
        )

    def tokenize_npz(self, file_path, max_len=2048):
        ds = self.dataset_mod.GECToRDataset(
            file_list=[file_path],
            data_dir=os.path.dirname(file_path) if os.path.isabs(file_path) else DATA_DIR,
            max_len=max_len,
        )
        return ds._tokenize_npz(0)

    def load_model(self, device='cpu'):
        return self.inference_mod.load_model_for_inference(self.checkpoint_path, device)

    def inference_single(self, model, tokens, device='cpu'):
        return self.inference_mod.inference_single(
            model, tokens, device=device,
            max_iterations=self.max_iterations,
            keep_confidence_bias=0.3,
            error_threshold=0.5,
        )

    def parse_sequence(self, tokens):
        return self.sequence_parser.parse_sequence(tokens)

    def decode_beat(self, tokens):
        return self.sequence_parser.decode_beat(tokens)


# ==================== FELIX Loader ====================

class FELIXLoader:
    """Loads FELIX modules for a single scheme using module isolation."""

    def __init__(self, scheme):
        self.scheme = scheme
        info = SCHEME_INFO[scheme]
        self.felix_dir = info['felix_dir']

        # Load FELIX modules with isolation
        saved_modules = _save_felix_modules()
        saved_path = sys.path[:]
        sys.path.insert(0, self.felix_dir)

        try:
            # Import modules fresh for this scheme
            from data.dataset import FELIXBaseDataset
            from data.perturbation import perturb_accompaniment
            from data.sequence_parser import (
                parse_sequence, separate_tracks, rebuild_interleaved,
                decode_beat, encode_beat,
            )
            from inference.pipeline import FELIXPipeline

            # Store references
            self._FELIXBaseDataset = FELIXBaseDataset
            self._perturb_accompaniment = perturb_accompaniment
            self._parse_sequence = parse_sequence
            self._separate_tracks = separate_tracks
            self._rebuild_interleaved = rebuild_interleaved
            self._decode_beat = decode_beat
            self._encode_beat = encode_beat
            self._FELIXPipeline = FELIXPipeline

            # Keep a snapshot of modules for later pipeline loading
            self._felix_modules = {}
            for k in list(sys.modules.keys()):
                if any(k == p or k.startswith(p + '.') for p in _FELIX_PKG_PREFIXES):
                    self._felix_modules[k] = sys.modules[k]

        finally:
            sys.path = saved_path
            _restore_felix_modules(saved_modules)

    def tokenize_npz(self, file_path, max_len=4096):
        file_name = os.path.basename(file_path) if os.path.isabs(file_path) else file_path
        data_dir = os.path.dirname(file_path) if os.path.isabs(file_path) else DATA_DIR
        ds = self._FELIXBaseDataset(
            file_list=[file_name], data_dir=data_dir,
            max_len=max_len, pitch_shift_augment=False,
        )
        return ds._tokenize_npz(0)

    def perturb_accompaniment(self, accomp_beats):
        return self._perturb_accompaniment(copy.deepcopy(accomp_beats))

    def parse_sequence(self, tokens):
        return self._parse_sequence(tokens)

    def separate_tracks(self, parsed):
        return self._separate_tracks(parsed)

    def rebuild_interleaved(self, melody_beats, accomp_beats, parsed):
        return self._rebuild_interleaved(melody_beats, accomp_beats, parsed)

    def decode_beat(self, tokens):
        return self._decode_beat(tokens)

    def load_pipeline(self, device='cuda'):
        """Load the FELIX (TagFill) Tagger+Inserter pipeline."""
        # NOTE: adjust to your checkpoint layout — TagFill checkpoints live under
        # checkpoints/tagfill/scheme_X/ in the release tree.
        tagfill_ckpt_dir = os.path.join(ROOT_DIR, 'checkpoints', 'tagfill', f'scheme_{self.scheme}')
        tagger_path = os.path.join(tagfill_ckpt_dir, 'tagger', 'tagger_best.pt')
        inserter_path = os.path.join(tagfill_ckpt_dir, 'inserter', 'inserter_best.pt')

        # Must restore FELIX modules for pipeline to load correctly
        saved_modules = _save_felix_modules()
        saved_path = sys.path[:]
        sys.path.insert(0, self.felix_dir)
        sys.modules.update(self._felix_modules)

        try:
            pipeline = self._FELIXPipeline(tagger_path, inserter_path, device=device)
        finally:
            sys.path = saved_path
            _restore_felix_modules(saved_modules)
            # Keep pipeline's internal references alive
        return pipeline

    def run_pipeline(self, pipeline, source_tokens, num_iterations=2):
        """Run FELIX pipeline with module isolation."""
        saved_modules = _save_felix_modules()
        saved_path = sys.path[:]
        sys.path.insert(0, self.felix_dir)
        sys.modules.update(self._felix_modules)

        try:
            result = pipeline.generate(source_tokens, num_iterations=num_iterations)
        finally:
            sys.path = saved_path
            _restore_felix_modules(saved_modules)
        return result

    def run_pipeline_inpainting(self, pipeline, token_sequence, mask_start_beat, mask_end_beat, mask_track=1):
        """Run FELIX inpainting with module isolation."""
        saved_modules = _save_felix_modules()
        saved_path = sys.path[:]
        sys.path.insert(0, self.felix_dir)
        sys.modules.update(self._felix_modules)

        try:
            result = pipeline.generate_inpainting(
                token_sequence, mask_start_beat, mask_end_beat, mask_track=mask_track
            )
        finally:
            sys.path = saved_path
            _restore_felix_modules(saved_modules)
        return result


# ==================== Unified Scheme Loader ====================

class SchemeLoader:
    """
    Unified loader for a single encoding scheme.
    Provides access to both GECToR and FELIX subsystems.
    """

    def __init__(self, scheme):
        assert scheme in SCHEME_INFO, f"Unknown scheme: {scheme}"
        self.scheme = scheme
        self.info = SCHEME_INFO[scheme]
        self._gector = None
        self._felix = None

    @property
    def gector(self):
        if self._gector is None:
            self._gector = GECToRLoader(self.scheme)
        return self._gector

    @property
    def felix(self):
        if self._felix is None:
            self._felix = FELIXLoader(self.scheme)
        return self._felix

    # -- Tokenization (uses FELIX dataset for consistency) --

    def tokenize_npz(self, file_path, max_len=2048):
        """Tokenize npz file, truncate to max_len."""
        tokens = self.felix.tokenize_npz(file_path, max_len=4096)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        return tokens

    # -- Parsing/decoding (scheme-specific) --

    def parse_sequence(self, tokens):
        return self.felix.parse_sequence(tokens)

    def separate_tracks(self, parsed):
        return self.felix.separate_tracks(parsed)

    def rebuild_interleaved(self, melody_beats, accomp_beats, parsed):
        return self.felix.rebuild_interleaved(melody_beats, accomp_beats, parsed)

    def decode_beat(self, tokens):
        return self.felix.decode_beat(tokens)

    # -- Perturbation --

    def perturb_accompaniment(self, accomp_beats):
        return self.felix.perturb_accompaniment(accomp_beats)


# ==================== Test file selection ====================

def get_test_files(data_dir=DATA_DIR, test_ratio=0.05, seed=42):
    """
    Get the fixed list of test files.
    Same logic as soft_eval_all.py get_test_files().
    """
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)
    test_size = int(len(all_files) * test_ratio)
    return [all_files[i] for i in indices[-test_size:]]
