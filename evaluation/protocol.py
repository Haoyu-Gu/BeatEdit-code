"""Scoring provenance and sample alignment shared by evaluation entry points."""
import hashlib
import json
import math
from pathlib import Path


METRIC_VERSION = 'beat-aligned-v2'


def scoring_protocol(metric_space='token'):
    if metric_space not in ('token', 'decoded'):
        raise ValueError('metric_space must be token or decoded')
    return {'version': METRIC_VERSION, 'beat_exact_space': metric_space,
            'empty_target_beats': 'include', 'unmatched_beats': 'penalize'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_record(test_path, pred_path, data, loader):
    parsed = loader.parse_sequence(data['target_tokens'])
    tracks = loader.separate_tracks(parsed)
    # Canonical targets allow paired comparisons across encoding schemes.
    target = {'tracks': [[sorted(loader.decode_beat(b['tokens'])) for b in track]
                         for track in tracks],
              'changed_beats': sorted(data['changed_beat_indices']),
              'level': data.get('level')}
    canonical = json.dumps(target, sort_keys=True, separators=(',', ':'))
    return {'sample_id': Path(test_path).name, 'test_sha256': file_hash(test_path),
            'prediction_sha256': file_hash(pred_path),
            'target_sha256': hashlib.sha256(canonical.encode()).hexdigest()}


class SampleScores(list):
    """Numeric scores retaining their ordered sample identities for pairing."""
    def __init__(self, values, records, protocol, scope, scheme=None):
        super().__init__(values)
        self.records = records
        self.protocol = protocol
        self.scope = scope
        self.scheme = scheme


def require_paired_samples(a, b):
    if not isinstance(a, SampleScores) or not isinstance(b, SampleScores):
        raise ValueError('Paired comparisons require sample identities; recompute scores')
    identity = lambda scores: [(r['sample_id'], r['target_sha256']) for r in scores.records]
    if (identity(a) != identity(b) or a.protocol != b.protocol or a.scope != b.scope):
        raise ValueError('Paired comparisons require identical ordered targets and scoring protocol')
    if a.scheme is not None and a.scheme == b.scheme:
        if [r['test_sha256'] for r in a.records] != [r['test_sha256'] for r in b.records]:
            raise ValueError('Paired comparisons require the same input test cases')


def scores_from_result(data, task, method, scheme, scope, metric_space='token',
                       test_data_dir=None, predictions_dir=None):
    """Reject legacy/mismatched caches instead of silently reusing their numbers."""
    expected = dict(task=task, method=method, scheme=scheme, scope=scope)
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError('Result metadata does not match the requested evaluation')
    protocol = scoring_protocol(metric_space)
    if data.get('scoring_protocol') != protocol:
        raise ValueError('Scoring protocol changed or is missing; rerun evaluate.py --per_sample')
    if data.get('num_errors') != 0:
        raise ValueError('Statistical tests require a complete evaluation')
    records = data.get('samples', [])
    ids = [r['sample_id'] for r in records]
    if (not records or len(records) != data.get('num_samples')
            or len(set(ids)) != len(ids) or ids != sorted(ids)):
        raise ValueError('Missing, duplicate, or unordered sample identities; recompute scores')
    for record in records:
        if Path(record['sample_id']).name != record['sample_id']:
            raise ValueError('Sample identity must be a filename')
        for key in ('test_sha256', 'prediction_sha256', 'target_sha256'):
            digest = record.get(key, '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Missing input fingerprint; recompute scores')
    # When raw inputs are explicitly supplied, verify the cache against them.
    for directory, key in [(test_data_dir, 'test_sha256'),
                           (predictions_dir, 'prediction_sha256')]:
        if directory is None:
            continue
        folder = Path(directory) / scheme if key == 'test_sha256' else Path(directory) / method / scheme
        if key == 'test_sha256' and sorted(p.name for p in folder.glob('*.json')) != ids:
            raise ValueError('Cached sample set does not match the test directory')
        for record in records:
            path = folder / record['sample_id']
            if not path.is_file() or file_hash(path) != record[key]:
                raise ValueError(f'Cached scores do not match input: {path}')
    by_id = {r['sample_id']: r for r in records}
    result = {}
    for metric, entry in data.get('overall', {}).items():
        if 'per_sample' not in entry:
            continue
        values = entry['per_sample']
        metric_ids = entry.get('sample_ids', [])
        if (len(values) != len(metric_ids) or len(values) != entry.get('n')
                or len(set(metric_ids)) != len(metric_ids)
                or metric_ids != [sid for sid in ids if sid in metric_ids]
                or not all(math.isfinite(v) for v in values)):
            raise ValueError(f'Invalid per-sample identities or scores for {metric}')
        result[metric] = SampleScores(values, [by_id[sid] for sid in metric_ids], protocol, scope, scheme)
    return result or None
