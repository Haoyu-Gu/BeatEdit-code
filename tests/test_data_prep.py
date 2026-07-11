#!/usr/bin/env python3
"""End-to-end test of the preprocessing chain (needs music21 + pretty_midi).

Synthesizes a two-staff piano score with known contents, converts it with
``data_prep/xml2npz.py``, and checks that

  * the npz layout matches what ``PianoDataset`` expects (4 channels, 88 keys
    ascending from A0, tau = 4 steps per beat, the metadata schema);
  * ``--velocity`` yields the 6-channel variant;
  * the pitches survive the round trip score -> npz -> tokens -> MIDI.

Run: python tests/test_data_prep.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TREBLE_PITCHES = [60, 64, 67, 72]     # C4 E4 G4 C5
BASS_PITCH = 48                       # C3
NUM_MEASURES = 20

REQUIRED_METADATA = {
    'time_signature', 'time_signature_idx', 'key_signature', 'key_signature_idx',
    'bpm', 'tempo_text', 'num_measures', 'resolution', 'total_length',
    'num_parts', 'num_channels', 'original_measures', 'valid_measures',
}


def _write_test_score(path):
    import music21 as m21
    score = m21.stream.Score()
    treble, bass = m21.stream.Part(), m21.stream.Part()
    for m in range(NUM_MEASURES):
        tm, bm = m21.stream.Measure(number=m + 1), m21.stream.Measure(number=m + 1)
        if m == 0:
            for stream in (tm, bm):
                stream.insert(0, m21.meter.TimeSignature('4/4'))
            tm.insert(0, m21.tempo.MetronomeMark(number=120))
            tm.insert(0, m21.key.Key('C'))
        for i in range(4):
            tm.append(m21.note.Note(TREBLE_PITCHES[(m + i) % 4], quarterLength=1.0))
        bm.append(m21.note.Note(BASS_PITCH, quarterLength=4.0))
        treble.append(tm)
        bass.append(bm)
    score.insert(0, treble)
    score.insert(0, bass)
    score.write('musicxml', fp=path)


def _convert(xml_path, out_dir, velocity=False):
    cmd = [sys.executable, os.path.join(ROOT, 'data_prep', 'xml2npz.py'),
           xml_path, '--output-dir', out_dir]
    if velocity:
        cmd.append('--velocity')
    subprocess.run(cmd, check=True, capture_output=True)
    return os.path.join(out_dir, os.path.basename(xml_path).rsplit('.', 1)[0] + '.npz')


def test_npz_layout(tmp):
    xml = os.path.join(tmp, 'piece.xml')
    _write_test_score(xml)
    npz = np.load(_convert(xml, os.path.join(tmp, 'npz')), allow_pickle=True)
    meta = npz['metadata'].item()

    assert REQUIRED_METADATA.issubset(meta), REQUIRED_METADATA - set(meta)
    assert meta['time_signature_idx'] == 0                      # 4/4
    assert meta['bpm'] == 120.0
    assert meta['num_measures'] == NUM_MEASURES
    assert meta['num_channels'] == 4
    assert meta['resolution'] == 16                             # 4 beats x tau=4

    m0 = npz['measure_0']
    assert m0.shape == (4, 88, 16), m0.shape
    assert m0.dtype == np.uint8
    assert set(np.unique(m0)) <= {0, 1}

    # pitch axis ascending from A0: row = MIDI - 21
    treble_rows = np.where(m0[0].any(axis=1))[0]
    bass_rows = np.where(m0[2].any(axis=1))[0]
    assert sorted(21 + treble_rows) == sorted(TREBLE_PITCHES), 21 + treble_rows
    assert list(21 + bass_rows) == [BASS_PITCH], 21 + bass_rows
    assert m0[1].sum() == 4          # four quarter-note onsets in the treble
    assert m0[2].sum() == 16         # the whole note fills the bass measure


def test_velocity_variant(tmp):
    xml = os.path.join(tmp, 'piece.xml')
    if not os.path.exists(xml):
        _write_test_score(xml)
    npz = np.load(_convert(xml, os.path.join(tmp, 'npz_vel'), velocity=True), allow_pickle=True)
    assert npz['metadata'].item()['num_channels'] == 6
    m0 = npz['measure_0']
    assert m0.shape == (6, 88, 16), m0.shape
    assert m0[2].max() > 0           # treble velocity channel is populated


def test_roundtrip_to_midi(tmp):
    import pretty_midi
    sys.path.insert(0, os.path.join(ROOT, 'src', 'encoding', 'scheme_A'))
    from config import ModelConfig
    from my_tokenizer import PianoRollTokenizer
    from PianoDataset import PianoDataset
    from token2midi import Token2MIDI

    xml = os.path.join(tmp, 'piece.xml')
    if not os.path.exists(xml):
        _write_test_score(xml)
    npz_dir = os.path.join(tmp, 'npz')
    if not os.path.isdir(npz_dir):
        _convert(xml, npz_dir)

    cfg = ModelConfig()
    dataset = PianoDataset(npz_dir, cfg, cache_lengths=False, mode='train',
                           test_split_ratio=0.0)
    tokens = dataset[0]['input_ids'].numpy()

    tokenizer = PianoRollTokenizer(patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                                   pattern_num=cfg.pattern_num,
                                   beats_length=cfg.beats_length)
    midi_path = os.path.join(tmp, 'roundtrip.mid')
    Token2MIDI(tokenizer, cfg).convert(tokens, midi_path)

    midi = pretty_midi.PrettyMIDI(midi_path)
    voices = [sorted({n.pitch for n in inst.notes}) for inst in midi.instruments if inst.notes]

    # PianoDataset transposes by +-5 semitones with p=0.7 (paper: data
    # augmentation), so the pitches survive only up to a constant offset.
    shift = voices[0][0] - min(TREBLE_PITCHES)
    assert abs(shift) <= 5, f"transposition {shift} outside the augmentation range"
    assert voices[0] == sorted(p + shift for p in TREBLE_PITCHES), voices[0]
    assert voices[1] == [BASS_PITCH + shift], voices[1]


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in sorted(globals().items()):
            if name.startswith('test_'):
                fn(tmp)
                print(f"{name}: PASS")
    print("All data_prep tests passed.")
