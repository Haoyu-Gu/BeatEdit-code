"""CPU regression tests using synthetic beats; no checkpoints or dataset needed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
sys.path.insert(0, str(ROOT / 'tools'))

import metrics
import statistical_tests
import anova_and_pairwise as anova
import evaluate
import reeval_decoded_beat_exact as decoded
from scheme_utils import SchemeLoader
from encoding_demo import beats_to_pianoroll


def beat(pitch=39, pattern=53):
    return {'tokens': [(pitch, pattern)]}


def decode(tokens):
    return tokens


class EvaluationTests(unittest.TestCase):
    def test_length_mismatch_in_both_directions(self):
        for pred, target in [([beat()], [beat(), beat(43)]),
                             ([beat(), beat(43)], [beat()])]:
            with self.subTest(pred=pred):
                result = metrics.compute_all_metrics(pred, target, decode)
                self.assertEqual(result['beat_exact_match'], .5)
                self.assertAlmostEqual(result['note_f1_tol0'], 2 / 3)
                self.assertAlmostEqual(result['chroma_f1'], 2 / 3)
                self.assertEqual(result['mean_pitch_error'], 22)
                self.assertEqual(result['token_accuracy'], .5)
                self.assertEqual(decoded.decoded_beat_exact_match(pred, target, decode), .5)
                self.assertEqual(metrics.context_preservation(pred, target, [False, False]), .5)

    def test_missing_empty_beat_is_not_a_match(self):
        empty = {'tokens': []}
        self.assertEqual(metrics.beat_exact_match([], [empty]), 0)
        self.assertEqual(metrics.token_accuracy([], [empty], decode), 0)
        self.assertEqual(decoded.decoded_beat_exact_match([], [empty], decode), 0)
        self.assertEqual(metrics.beat_exact_match([empty], [empty]), 1)
        with self.assertRaises(ValueError):
            metrics.context_preservation([], [empty], [])

    def test_scoped_alignment_retains_missing_and_extra_beats(self):
        pred, target = metrics.align_beats([beat()], [beat(), beat(43)], {1})
        self.assertEqual(pred, [None])
        self.assertEqual(metrics.beat_exact_match(pred, target), 0)
        pred, target = metrics.align_beats([beat(), beat(43)], [beat()], {0})
        self.assertEqual(target[-1], None)
        self.assertEqual(metrics.beat_exact_match(pred, target), .5)

    def test_equal_length_semantics_are_preserved(self):
        same = [beat(), beat(43)]
        result = metrics.compute_all_metrics(same, same, decode)
        for key, value in result.items():
            self.assertEqual(value, 0 if key == 'mean_pitch_error' else 1, key)
        # Note F1 remains pitch-based, whereas BME includes the pattern.
        result = metrics.compute_all_metrics([beat(pattern=50)], [beat()], decode)
        self.assertEqual(result['note_f1_tol0'], 1)
        self.assertEqual(result['beat_exact_match'], 0)
        # Rhythm similarity remains distributional but must include tail beats.
        self.assertAlmostEqual(metrics.rhythm_similarity(
            [beat()], [beat(), beat(pattern=50)], decode), 1 / np.sqrt(2))

    def test_public_evaluation_and_anova_loading(self):
        # Real Scheme A parser and pred_tokens file format, including short output.
        target = [172, 174, 181, 170, 183, 120, 53, 184, 120, 53,
                  183, 124, 53, 184, 124, 53, 171]
        pred = [172, 174, 181, 170, 183, 120, 53, 184, 120, 53, 171]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tests = base / 'test_data' / 'editing'
            predictions = base / 'predictions' / 'editing'
            results = base / 'results'
            (tests / 'A').mkdir(parents=True)
            (predictions / 'gector' / 'A').mkdir(parents=True)
            results.mkdir()
            test_path = tests / 'A' / 'sample.json'
            pred_path = predictions / 'gector' / 'A' / 'sample.json'
            test_path.write_text(json.dumps({'target_tokens': target, 'level': 'easy',
                                            'changed_beat_indices': [1]}))
            pred_path.write_text(json.dumps({'pred_tokens': pred}))
            for scope, expected in [('perturbed_only', 0), ('full_sequence', .5)]:
                result = evaluate.evaluate_method_scheme(
                    'gector', 'A', str(tests), str(predictions), scope=scope,
                    include_per_sample=True, include_ci=False)
                self.assertEqual(result['num_errors'], 0)
                self.assertEqual(result['overall']['beat_exact_match']['mean'], expected)
                old, new = decoded.evaluate_one(test_path, pred_path, SchemeLoader('A'), scope)
                self.assertEqual((old, new), (expected, expected))
                scores = anova.load_per_sample_scores(
                    'editing', 'gector', 'A', scope, results_dir=str(results),
                    test_data_dir=str(tests), predictions_dir=str(predictions))
                self.assertEqual(scores, [expected])  # Raw-file fallback.
                output = results / f'editing_gector_A_{scope}.json'
                output.write_text(json.dumps(result))
                self.assertEqual(anova.load_per_sample_scores(
                    'editing', 'gector', 'A', scope, results_dir=str(results)), [expected])
                result['num_errors'] = 1
                output.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    anova.load_per_sample_scores(
                        'editing', 'gector', 'A', scope, results_dir=str(results))
            # Decoded re-evaluation must also count absent prediction files.
            (tests / 'A' / 'missing.json').write_text(test_path.read_text())
            with patch.object(decoded, 'UNIFIED_DIR', str(base)):
                result = decoded.evaluate_method_scheme('editing', 'gector', 'A')
                self.assertEqual(result['n'], 1)
                self.assertEqual(result['errors'], 1)
            pred_path.unlink()
            with self.assertRaises(FileNotFoundError):
                statistical_tests.compute_per_sample_scores(
                    'editing', 'gector', 'A', test_data_dir=str(tests),
                    predictions_dir=str(predictions))

    def test_balanced_anova_and_invalid_designs(self):
        methods, schemes = ['a', 'b'], ['A', 'B']
        data = {('a', 'A'): [1, 2], ('a', 'B'): [3, 4],
                ('b', 'A'): [5, 6], ('b', 'B'): [8, 9]}
        result = anova.two_way_anova(data, methods, schemes)
        self.assertEqual(result['Method']['df'], 1)
        self.assertEqual(result['Encoding']['df'], 1)
        self.assertEqual(result['Error']['df'], 4)
        self.assertEqual(result['Error']['MS'], .5)
        self.assertEqual(result['Method']['SS'], 40.5)
        self.assertEqual(result['Encoding']['SS'], 12.5)
        self.assertEqual(result['Method×Encoding']['SS'], .5)
        for name in ['Method', 'Encoding', 'Method×Encoding']:
            self.assertAlmostEqual(result[name]['F'], result[name]['MS'] / .5)
        for cell in [[], [1, 2, 3], [1, float('nan')]]:
            with self.assertRaises(ValueError):
                anova.two_way_anova({**data, ('a', 'A'): cell}, methods, schemes)
        constant_cells = {key: [values[0]] * 2 for key, values in data.items()}
        zero_error = anova.two_way_anova(constant_cells, methods, schemes)
        self.assertEqual(zero_error['Method']['F'], float('inf'))
        self.assertEqual(zero_error['Method']['p'], 0)

    def test_pairwise_refuses_truncated_arrays(self):
        with self.assertRaises(ValueError):
            anova.paired_bootstrap([1, 2], [1])
        with patch.object(statistical_tests, 'get_per_sample_scores',
                          side_effect=[{'beat_exact_match': [1, 2]},
                                       {'beat_exact_match': [1]}]):
            with self.assertRaises(ValueError):
                statistical_tests.compare_methods('editing', 'a', 'b', ['A'])

    def test_midi_repeated_onsets_and_cross_beat_sustain_all_schemes(self):
        for scheme in 'ABCD':
            directory = ROOT / 'src' / 'encoding' / f'scheme_{scheme}'
            # Converters use bare local imports; isolate each scheme's bindings.
            saved_path = sys.path[:]
            saved = {key: sys.modules.pop(key, None) for key in ('config', 'my_tokenizer')}
            try:
                sys.path.insert(0, str(directory))
                spec = importlib.util.spec_from_file_location(
                    f'converter_{scheme}', directory / 'token2midi.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                converter = module.Token2MIDI.__new__(module.Token2MIDI)
                for beats, expected in [
                        ([[(39, 50)]], [(0, .25), (.25, .5)]),
                        ([[(39, 53)], [(39, 80)]], [(0, 1)]),
                        ([[(39, 53)], [(39, 53)]], [(0, .5), (.5, 1)])]:
                    roll = beats_to_pianoroll(beats)
                    track = converter._create_midi_track(roll[0], roll[1], .125, 64)
                    self.assertEqual([(n.start, n.end) for n in track.notes], expected, scheme)
                    self.assertTrue(all(n.pitch == 60 for n in track.notes))
            finally:
                sys.path = saved_path
                for key, value in saved.items():
                    sys.modules.pop(key, None)
                    if value is not None:
                        sys.modules[key] = value


if __name__ == '__main__':
    unittest.main()
