"""CPU regression checks for MLM split logic and the table-generation CLI."""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ReleaseProtocolTests(unittest.TestCase):
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
