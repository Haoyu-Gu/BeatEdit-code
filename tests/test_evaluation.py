"""CPU regression tests using synthetic beats; no checkpoints or dataset needed."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
sys.path.insert(0, str(ROOT / 'tools'))

import metrics
import statistical_tests
import anova_and_pairwise as anova
import summarize
import evaluate
import reeval_decoded_beat_exact as decoded
from scheme_utils import SchemeLoader, get_test_files
from encoding_demo import beats_to_pianoroll
from protocol import SampleScores, require_paired_samples, scoring_protocol


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

    def test_completion_denominator_and_scoring_spaces(self):
        target = [7133, 7135, 7141, 7131, 7129, 3212, 7128,
                  7129, 3536, 7130, 3212, 7132]
        pred = target[:-2] + [3536, 7132]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'test' / 'D').mkdir(parents=True)
            (base / 'pred' / 'felix' / 'D').mkdir(parents=True)
            tpath = base / 'test' / 'D' / 'sample.json'
            ppath = base / 'pred' / 'felix' / 'D' / 'sample.json'
            tpath.write_text(json.dumps(dict(target_tokens=target, level='gap',
                                            changed_beat_indices=[0, 1])))
            ppath.write_text(json.dumps(dict(pred_tokens=pred)))
            self.assertEqual(decoded.evaluate_one(
                tpath, ppath, SchemeLoader('D'), task='inpainting'), (.5, .5))
            for space in ('token', 'decoded'):
                result = evaluate.evaluate_method_scheme(
                    'felix', 'D', str(base / 'test'), str(base / 'pred'),
                    task='inpainting', metric_space=space,
                    include_ci=False, include_per_sample=True)
                scores = statistical_tests.compute_per_sample_scores(
                    'inpainting', 'felix', 'D', test_data_dir=str(base / 'test'),
                    predictions_dir=str(base / 'pred'), metric_space=space)
                self.assertEqual(result['overall']['beat_exact_match']['mean'], .5)
                self.assertEqual(scores['beat_exact_match'], [.5])
                self.assertEqual(scores['beat_exact_match'].records, result['samples'])
        # An explicitly empty perturbed region must not turn into full sequence.
        self.assertEqual(metrics.select_beats([beat()], [beat()], [], 'perturbed_only'), ([], []))
        reordered = [{'tokens': [(43, 53), (39, 53)]}]
        ordered = [{'tokens': [(39, 53), (43, 53)]}]
        self.assertEqual(metrics.compute_all_metrics(reordered, ordered, decode)['beat_exact_match'], 0)
        self.assertEqual(metrics.compute_all_metrics(
            reordered, ordered, decode, metric_space='decoded')['beat_exact_match'], 1)

    def test_cache_provenance_and_sample_alignment(self):
        from copy import deepcopy
        from protocol import scores_from_result
        record = dict(sample_id='a.json', test_sha256='a' * 64,
                      prediction_sha256='b' * 64, target_sha256='c' * 64)
        valid = dict(task='editing', method='felix', scheme='D', scope='perturbed_only',
                     scoring_protocol=scoring_protocol(), samples=[record],
                     num_samples=1, num_errors=0,
                     overall={'beat_exact_match': dict(n=1, per_sample=[.5], sample_ids=['a.json'])})
        load = lambda data, **kw: scores_from_result(
            data, 'editing', 'felix', 'D', 'perturbed_only', **kw)
        score = load(valid)['beat_exact_match']
        require_paired_samples(score, score)
        for key, value in [('task', 'correction'), ('scheme', 'C'), ('scope', 'full_sequence'),
                           ('method', 'gector'), ('scoring_protocol', {}), ('samples', []),
                           ('num_samples', 2), ('num_errors', 1)]:
            with self.subTest(field=key), self.assertRaises(ValueError):
                load(dict(valid, **{key: value}))
        with self.assertRaises(ValueError):
            load(valid, metric_space='decoded')
        for ids in ([], ['other.json']):
            broken = deepcopy(valid)
            broken['overall']['beat_exact_match']['sample_ids'] = ids
            with self.assertRaises(ValueError):
                load(broken)
        other = SampleScores([.5], [dict(record, target_sha256='d' * 64)],
                             scoring_protocol(), 'perturbed_only')
        with self.assertRaises(ValueError):
            require_paired_samples(score, other)
        with patch.object(statistical_tests, 'get_per_sample_scores',
                          side_effect=[{'beat_exact_match': score}, {'beat_exact_match': other}]):
            with self.assertRaises(ValueError):
                statistical_tests.compare_methods('editing', 'felix', 'gector', ['D'])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'D'
            folder.mkdir()
            (folder / 'a.json').write_text('{}')
            with self.assertRaises(ValueError):
                load(valid, test_data_dir=tmp)

    def test_summary_rejects_mixed_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = dict(sample_id='a.json', test_sha256='a' * 64,
                          prediction_sha256='b' * 64, target_sha256='c' * 64)
            for method, space in [('felix', 'token'), ('gector', 'decoded')]:
                data = dict(task='editing', method=method, scheme='D', scope='perturbed_only',
                            num_samples=1, num_errors=0, samples=[record], overall={},
                            scoring_protocol=scoring_protocol(space))
                Path(tmp, f'editing_{method}_D_perturbed_only.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                summarize.load_results(tmp)

    def test_evaluation_cli_to_paired_tests_and_summary(self):
        target = [7133, 7135, 7141, 7131, 7129, 3212, 7128,
                  7129, 3536, 7130, 3212, 7132]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            targets = base / 'test' / 'inpainting'
            predictions = base / 'pred'
            target_a = [172, 174, 181, 170, 183, 120, 53, 184, 0,
                        183, 124, 53, 184, 120, 53, 171]
            for scheme, reference in [('A', target_a), ('D', target)]:
                (targets / scheme).mkdir(parents=True)
                for method in ('felix', 'no_edit'):
                    (predictions / 'inpainting' / method / scheme).mkdir(parents=True)
                    for i in range(2):
                        (targets / scheme / f'{i}.json').write_text(json.dumps(dict(
                            target_tokens=reference, level='gap', changed_beat_indices=[0, 1])))
                        wrong = (target_a[:-3] + [124, 53, 171] if scheme == 'A'
                                 else target[:-2] + [3536, 7132])
                        pred = reference if method == 'felix' else wrong
                        (predictions / 'inpainting' / method / scheme / f'{i}.json').write_text(
                            json.dumps(dict(pred_tokens=pred)))
            results = base / 'results'
            def run(script, args):
                return subprocess.run([sys.executable, str(ROOT / 'evaluation' / script)] + args,
                                      cwd=ROOT, capture_output=True, text=True)
            evaluated = run('evaluate.py', ['--task', 'inpainting', '--schemes', 'A,D',
                '--methods', 'felix,no_edit', '--test_data_dir', str(targets),
                '--predictions_dir', str(predictions), '--output_dir', str(results),
                '--per_sample', '--metric_space', 'decoded', '--no_ci'])
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
            args = ['--task', 'inpainting', '--schemes', 'D', '--method_a', 'felix',
                '--method_b', 'no_edit', '--from_results', '--results_dir', str(results),
                '--metric_space', 'decoded', '--n_bootstrap', '20',
                '--test_data_dir', str(targets), '--predictions_dir', str(predictions / 'inpainting'),
                '--metrics', 'beat_exact_match']
            paired = run('statistical_tests.py', args)
            self.assertEqual(paired.returncode, 0, paired.stdout + paired.stderr)
            table = run('summarize.py', ['--task', 'inpainting', '--results_dir', str(results),
                                        '--output', str(base / 'table.md')])
            self.assertEqual(table.returncode, 0, table.stdout + table.stderr)
            self.assertIn('decoded', (base / 'table.md').read_text())
            anova_run = run('anova_and_pairwise.py', ['--task', 'inpainting',
                '--methods', 'felix,no_edit', '--schemes', 'A,D',
                '--results_dir', str(results), '--metric_space', 'decoded',
                '--pairwise', 'felix_A:felix_D'])
            self.assertEqual(anova_run.returncode, 0, anova_run.stdout + anova_run.stderr)
            output = json.loads((results / 'inpainting_anova_perturbed_only.json').read_text())
            self.assertEqual(output['cell_counts'], dict(felix_A=2, felix_D=2, no_edit_A=2, no_edit_D=2))
            # Reusing the same filenames is insufficient after input edits.
            path = targets / 'D' / '0.json'
            path.write_text(path.read_text() + '\n')
            rejected = run('statistical_tests.py', args)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Cached scores do not match input', rejected.stderr)

    def test_evaluation_test_partition_matches_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(100):
                Path(tmp, f'song_{i:04}.npz').touch()
            indices = np.arange(100)
            np.random.RandomState(42).shuffle(indices)
            self.assertEqual(get_test_files(tmp), [f'song_{i:04}.npz' for i in indices[90:]])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'one.npz').touch()
            with self.assertRaises(ValueError):
                get_test_files(tmp)

    def test_tagger_gap_completion_all_schemes(self):
        for scheme in 'ABCD':
            loader = SchemeLoader(scheme)
            cls = loader.felix._FELIXPipeline
            glob = cls.generate_inpainting.__wrapped__.__globals__
            parser, cfg = glob['_gap_parser'], glob['_gap_config']
            # Two beats per track, with a nonempty region that must be erased.
            def make_sequence(gap_notes):
                mel = [dict(tokens=parser.encode_beat([(39, 53)])) for _ in range(2)]
                acc = [dict(tokens=parser.encode_beat(gap_notes)),
                       dict(tokens=parser.encode_beat([(43, 53)]))]
                if scheme == 'A':
                    for b in mel:
                        b['track_id'] = cfg.TRACK0_START
                    for b in acc:
                        b['track_id'] = cfg.TRACK1_START
                elif scheme in 'CD':
                    for b in mel:
                        b['split_id'] = cfg.SPLIT_0
                    for b in acc:
                        b['split_id'] = cfg.SPLIT_1 if b['tokens'] else None
                info = dict(header_tokens=[cfg.BOS_TOKEN, cfg.TIME_SIG_OFFSET, cfg.BPM_OFFSET],
                            footer_tokens=[cfg.EOS_TOKEN], bars=[dict(beats=[{}, {}, {}, {}])])
                return parser.rebuild_interleaved(mel, acc, info)
            pipe = cls.__new__(cls)
            pipe.tagger = pipe.inserter = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=2048))
            tagged = []
            inserted = []
            expected_content = parser.encode_beat([(39, 53)])
            # B's END marker is predicted too; A/C/D allocate just content.
            n = len(expected_content)
            def tag(tokens):
                tagged.append(list(tokens))
                _, acc = parser.separate_tracks(parser.parse_sequence(tokens))
                labels = [cfg.LABEL_KEEP] * len(tokens)
                labels[acc[0]['end_idx'] - 1] = cfg.label_id_append(n)
                # Deliberately propose changes outside the gap: must be ignored.
                labels[0] = cfg.LABEL_DELETE
                return labels
            def insert(tokens, positions, *args):
                inserted.append(len(positions))
                self.assertEqual(len(positions), n)
                output = list(tokens)
                for position, token in zip(positions, expected_content):
                    output[position] = token
                return output
            pipe._run_tagger = tag
            pipe._run_inserter = pipe._run_inserter_iterative = insert
            outputs = []
            for notes, steps in [([], 1), ([(43, 53), (47, 53)], 2)]:
                seq = make_sequence(notes)
                out = pipe.generate_inpainting(seq, 0, 1, inserter_steps=steps)
                outputs.append(out)
                before = parser.separate_tracks(parser.parse_sequence(seq))
                after = parser.separate_tracks(parser.parse_sequence(out))
                self.assertEqual([b['tokens'] for b in before[0]],
                                 [b['tokens'] for b in after[0]], scheme)
                self.assertEqual(before[1][1]['tokens'], after[1][1]['tokens'], scheme)
                self.assertEqual(parser.decode_beat(after[1][0]['tokens']), [(39, 53)], scheme)
                left, right = before[1][0]['start_idx'], before[1][0]['end_idx']
                new_left, new_right = after[1][0]['start_idx'], after[1][0]['end_idx']
                self.assertEqual(seq[:left], out[:new_left], scheme)
                self.assertEqual(seq[right:], out[new_right:], scheme)
            self.assertEqual(tagged[0], tagged[1], scheme)  # independent of old length/content
            self.assertEqual(outputs[0], outputs[1], scheme)
            self.assertEqual(inserted, [n, n], scheme)
            pipe._run_tagger = lambda tokens: [cfg.LABEL_KEEP] * len(tokens)
            empty_result = pipe.generate_inpainting(make_sequence([(39, 53)]), 0, 1)
            _, empty_acc = parser.separate_tracks(parser.parse_sequence(empty_result))
            self.assertEqual(parser.decode_beat(empty_acc[0]['tokens']), [], scheme)
            for start, end, track in [(-1, 1, 1), (0, 3, 1), (0, 1, 4)]:
                with self.assertRaises(ValueError):
                    pipe.generate_inpainting(make_sequence([]), start, end, track)
            pipe.tagger.config.max_position_embeddings = 1
            with self.assertRaises(ValueError):
                pipe.generate_inpainting(make_sequence([]), 0, 1)

    def test_gap_completion_tiny_neural_forward(self):
        import torch
        loader = SchemeLoader('D')
        cls = loader.felix._FELIXPipeline
        glob = cls.generate_inpainting.__wrapped__.__globals__
        cfg = glob['_gap_config']
        small = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64, dropout=0.0)
        tagger = glob['FELIXTagger'](cfg.TaggerConfig(**small))
        inserter = glob['FELIXInserter'](cfg.InserterConfig(**small))
        with torch.no_grad():
            tagger.classifier.weight.zero_()
            tagger.classifier.bias.fill_(-20)
            tagger.classifier.bias[cfg.label_id_append(1)] = 20
            inserter.mlm_head.weight.zero_()
            inserter.mlm_head.bias.fill_(-20)
            inserter.mlm_head.bias[3212] = 20
        pipe = cls(tagger, inserter, device='cpu')
        calls = []
        handles = [model.register_forward_hook(lambda *args, name=name: calls.append(name))
                   for name, model in [('tagger', tagger), ('inserter', inserter)]]
        tokens = [7133, 7135, 7141, 7131, 7129, 3536, 7128, 7132]
        try:
            out = pipe.generate_inpainting(tokens, 0, 1, inserter_steps=1)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, ['tagger', 'inserter'])
        self.assertEqual(out, tokens[:-2] + [7130, 3212, 7132])

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
