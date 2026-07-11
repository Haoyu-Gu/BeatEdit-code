# Data preparation

Converts symbolic scores into the `.npz` piano-roll format that
`src/encoding/scheme_*/PianoDataset.py` (and every training script downstream)
reads.

```bash
# MuseScore MusicXML corpus -> npz  (the main piano experiments)
python data_prep/xml2npz.py  /path/to/musescore/ --output-dir /path/to/data/npz --workers 8

# MIDI sources (e.g. Lakh) -> npz
python data_prep/midi2npz.py /path/to/lakh/     --output-dir /path/to/data/npz --workers 8

# then point the training scripts at it
export BEATEDIT_DATA_DIR=/path/to/data/npz
```

## Output format

One `.npz` per piece:

| key | dtype / shape | meaning |
|-----|---------------|---------|
| `measure_{i}` | `uint8 (C, 88, T)` | one array per measure |
| `metadata` | pickled dict | see below |

* **Channels** `C = 4` by default: `[treble_sus, treble_ons, bass_sus, bass_ons]`.
  With `--velocity`, `C = 6`: `[treble_sus, treble_ons, treble_vel, bass_sus,
  bass_ons, bass_vel]`.
* **Pitch axis** — index `0` = MIDI 21 (A0) … index `87` = MIDI 108 (C8), ascending.
* **Time axis** — `T = 4 x (quarter notes per measure)`, i.e. the sixteenth-note
  grid (`tau = 4` in the paper). A 4/4 measure is 16 steps, 3/4 is 12.
* `sus` / `ons` are binary; `vel` holds MIDI velocity 0–127.

`metadata` keys: `time_signature`, `time_signature_idx` (0=4/4, 1=3/4, 2=2/4,
3=6/8, 4=2/2), `key_signature`, `key_signature_idx` (number of sharps; flats
negative), `bpm`, `tempo_text`, `num_measures`, `resolution` (steps per
measure), `total_length`, `num_parts`, `num_channels`, `original_measures`,
`valid_measures`, `is_continuation`.

The sustain/onset pair is what the ternary Beat Encoding is built from
(`0` = silent, `1` = onset, `2` = sustain continuation — paper Appendix A).

## Velocity

Velocity is **off by default**, matching the paper: the reported experiments use
the 4-channel pitch/rhythm representation, and velocity is orthogonal to the
editing mechanism (an extra pattern channel edited by the same operators).
`--velocity` produces the 6-channel arrays for anyone who wants to model it;
`PianoRollTokenizer` consumes them when constructed with `use_velocities=True`
(see `compress_tokens_velocity` / `patch_tokens_to_image_velocity`).

## Filtering

A piece is skipped when it has fewer than 16 or more than 300 measures, has
fewer than two staves, or carries a time signature outside
{4/4, 3/4, 2/4, 6/8, 2/2}. Individual measures whose length does not match the
piece's time signature (pickup bars, irregular bars) are dropped; the indices
that survive are recorded in `metadata['valid_measures']`.

## Provenance and caveat

These converters are adapted from the BEAT reference implementation
([Lekai-Qian/BEAT-code](https://github.com/Lekai-Qian/BEAT-code),
`data_prep/xml2pianonpz.py`), retargeted to the BeatEdit layout (4 channels,
`tau = 4` steps per beat instead of BEAT's 24-tick grid).

They are a faithful *reimplementation* of the preprocessing used for the paper,
validated field-by-field against the `.npz` files the reported models were
trained on (identical `metadata` schema, array shapes, dtype and pitch-axis
orientation) and end-to-end through the encoder and decoder. They are not the
byte-for-byte original scripts, so edge cases (unusual tuplets, mid-piece time
signature changes) may quantize slightly differently; re-preprocessing the
corpus from scratch can therefore shift the last digits of the reported numbers.
Pre-computed results in `results/` come from the original preprocessing run.
