"""CPU regression checks for MLM split logic and the table-generation CLI."""
import ast
import contextlib
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ReleaseProtocolTests(unittest.TestCase):
    def test_evaluation_script_cli_contract(self):
        script = (ROOT / 'scripts/06_evaluate_all.sh').read_text().replace('\\\n', ' ')
        checked = 0
        for line in script.splitlines():
            if not line.strip().startswith('"$PYTHON"'):
                continue
            tokens = shlex.split(line)
            path = ROOT / tokens[1].replace('$REPO_DIR/', '')
            tree = ast.parse(path.read_text())
            allowed = {arg.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                for arg in n.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            passed = {t for t in tokens[2:] if t.startswith('--')}
            self.assertFalse(passed - allowed, f'{path}: {passed - allowed}')
            checked += 1
        self.assertEqual(checked, 2)

    def test_linear_prediction_heads(self):
        import torch
        from torch import nn
        cases = []
        for scheme in 'ABCD':
            base = ROOT / f'src/tagfill/scheme_{scheme}/models'
            cases += [(base / 'tagger.py', 'classifier', 512, 11),
                      (base / 'inserter.py', 'mlm_head', 512, 186 if scheme == 'A' else 185 if scheme == 'B' else 7145)]
        base = ROOT / 'src/iteredit/models/levenshtein_transformer.py'
        cases += [(base, 'del_head', 512, 2), (base, 'ins_head', 1024, 21),
                  (base, 'tok_head', 512, 7145)]
        for path, name, in_dim, out_dim in cases:
            tree = ast.parse(path.read_text())
            assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == name for t in n.targets))
            obj = SimpleNamespace()
            config = SimpleNamespace(hidden_size=512, num_labels=11, vocab_size=out_dim, max_insert=20)
            namespace = {'nn': nn, 'self': obj, 'config': config, 'hidden_size': 512}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), 'exec'), namespace)
            head = getattr(obj, name)
            self.assertIsInstance(head, nn.Linear)
            output = head(torch.zeros(2, 3, in_dim))
            self.assertEqual(tuple(output.shape), (2, 3, out_dim))
            output.sum().backward()
            self.assertIsNotNone(head.weight.grad)

    def test_downstream_partitions_match_mlm(self):
        paths = list((ROOT / 'src').rglob('dataset.py'))
        paths += [ROOT / 'src/iteredit/data/dataset_editing.py']
        expected = np.arange(100)
        np.random.RandomState(42).shuffle(expected)
        expected = [list(expected[:80]), list(expected[80:90]), list(expected[90:])]
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(100):
                for ext in ('npz', 'mid'):
                    Path(tmp, f'song_{i:04}.{ext}').touch()
            for path in paths:
                tree = ast.parse(path.read_text())
                methods = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name == 'get_file_lists']
                if not methods:
                    continue
                namespace = {'np': np, 'os': os, 'DATA_DIR': tmp, 'MIDI_DATA_DIR': tmp}
                exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
                result = namespace['get_file_lists'](tmp)
                ids = [[int(Path(f).stem.split('_')[1]) for f in part] for part in result]
                self.assertEqual(ids, expected, str(path))
                with self.assertRaises(ValueError):
                    namespace['get_file_lists'](tmp, test_ratio=0)
                checked += 1
            self.assertEqual(checked, 12)

    def test_legacy_encoding_partitions(self):
        for scheme in 'ABCD':
            path = ROOT / f'src/encoding/scheme_{scheme}/PianoDataset.py'
            tree = ast.parse(path.read_text())
            method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == '_split_train_test')
            namespace = {'np': np}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
            for mode, count in [('train', 80), ('validation', 10), ('test', 10)]:
                obj = SimpleNamespace(data_files=[f'song_{i:04}.npz' for i in reversed(range(100))],
                    file_lengths=None, config=SimpleNamespace(min_length=0), mode=mode,
                    random_seed=42, validation_split_ratio=.1, test_split_ratio=.1)
                namespace['_split_train_test'](obj)
                self.assertEqual(len(obj.data_files), count)

    def test_mlm_partitions(self):
        # Execute the actual split method without loading Torch/tokenizers.
        expected = None
        for scheme in 'ABCD':
            directory = ROOT / 'src/pretraining' / f'scheme_{scheme}'
            tree = ast.parse((directory / 'mlm_dataset.py').read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'MLMDataset')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_split_data')
            namespace = {'np': np}
            exec(compile(ast.Module(body=[method], type_ignores=[]),
                         str(directory / 'mlm_dataset.py'), 'exec'), namespace)
            partitions = []
            for mode in ('train', 'validation', 'test'):
                sets = []
                for reverse in (False, True):
                    names = [f'song_{i:04}.npz' for i in range(100)]
                    lengths = list(range(100))
                    if reverse:
                        names.reverse()
                        lengths.reverse()
                    obj = SimpleNamespace(mode=mode, data_files=names,
                        file_lengths=lengths, sorted_indices=None,
                        bc=SimpleNamespace(random_seed=42,
                            validation_split_ratio=.1, test_split_ratio=.1))
                    with contextlib.redirect_stdout(io.StringIO()):
                        namespace['_split_data'](obj)
                    self.assertEqual(obj.file_lengths,
                        [int(Path(f).stem.split('_')[1]) for f in obj.data_files])
                    sets.append(set(obj.data_files))
                self.assertEqual(*sets)
                partitions.append(sets[0])
            self.assertEqual([len(s) for s in partitions], [80, 10, 10])
            self.assertEqual(len(set.union(*partitions)), 100)
            for i in range(3):
                for j in range(i):
                    self.assertFalse(partitions[i] & partitions[j])
            if expected is not None:
                self.assertEqual(partitions, expected)
            expected = partitions
            calls = [n for n in ast.walk(ast.parse(
                (directory / 'train_mlm.py').read_text()))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'MLMDataset']
            modes = [kw.value.value for c in calls for kw in c.keywords
                     if kw.arg == 'mode']
            self.assertEqual(modes, ['train', 'validation'])

    def test_table_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs = base / 'inputs'
            inputs.mkdir()
            output = base / 'tables'
            command = ['bash', str(ROOT / 'scripts/07_generate_tables.sh'),
                       str(inputs), str(output)]
            empty = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(empty.returncode, 0)
            self.assertIn('No results found', empty.stderr)
            self.assertFalse(output.exists())
            for task in ('correction', 'editing', 'inpainting'):
                fixture = {'method': 'gector', 'scheme': 'A', 'task': task,
                           'overall': {'beat_exact_match': {'mean': .5}},
                           'per_level': {}}
                (inputs / f'{task}_gector_A_perturbed_only.json').write_text(
                    json.dumps(fixture))
            run = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            tables = list(output.glob('*.md'))
            self.assertEqual(len(tables), 3)
            for table in tables:
                self.assertIn('0.500', table.read_text())


if __name__ == '__main__':
    unittest.main()
