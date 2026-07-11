"""
FELIX-Music Token-Level Label Extractor.

Aligns source (perturbed) and target (clean) token sequences beat-by-beat,
extracting per-token edit labels from the 11-label space.

Like GECToR: one label per source token.
  KEEP=0, DELETE=1, REPLACE=2, APPEND_1..8 = 3..10

Label extraction uses beat-level alignment (structure is preserved by
perturbation), then assigns per-token labels within each beat.
"""

from configs.config import (
    LABEL_KEEP, LABEL_DELETE, LABEL_REPLACE,
    label_id_append, LABEL_APPEND_MAX_N,
    EMPTY_MARKER, PATTERN_NUM, BUNDLED_TOKEN_MAX,
    is_bundled_token,
)
from data.sequence_parser import parse_sequence, decode_beat, encode_beat


def extract_token_labels(source_tokens, target_tokens):
    """
    Extract per-token FELIX labels by aligning source and target beat-by-beat.

    Args:
        source_tokens: perturbed token sequence (list of ints)
        target_tokens: clean token sequence (list of ints)

    Returns:
        labels: list of label IDs, same length as source_tokens
        targets: list of lists, same length as source_tokens
            - For KEEP: []
            - For DELETE: []
            - For REPLACE: [target_token]
            - For APPEND_N: [target_token_1, ..., target_token_N]
    """
    source_info = parse_sequence(source_tokens)
    target_info = parse_sequence(target_tokens)

    labels = []
    targets = []

    # Header tokens → all KEEP
    for _ in source_info['header_tokens']:
        labels.append(LABEL_KEEP)
        targets.append([])

    # Process bars and beats
    s_beat_idx = 0
    for bar in source_info['bars']:
        # BAR token → KEEP
        labels.append(LABEL_KEEP)
        targets.append([])

        for beat in bar['beats']:
            if s_beat_idx < len(target_info['beats']):
                t_beat = target_info['beats'][s_beat_idx]
                beat_labels, beat_targets = _extract_beat_token_labels(beat, t_beat)
            else:
                beat_labels, beat_targets = _keep_labels_for_beat(beat)

            labels.extend(beat_labels)
            targets.extend(beat_targets)
            s_beat_idx += 1

    # Footer tokens → all KEEP
    for _ in source_info['footer_tokens']:
        labels.append(LABEL_KEEP)
        targets.append([])

    assert len(labels) == len(source_tokens), \
        f"Label length {len(labels)} != source length {len(source_tokens)}"
    assert len(targets) == len(source_tokens)

    return labels, targets


def _keep_labels_for_beat(beat):
    """Generate all-KEEP labels for a beat (including SPLIT/EMPTY marker)."""
    if beat['split_id'] is not None:
        n = 1 + len(beat['tokens'])  # SPLIT + bundled tokens
    else:
        n = 1  # EMPTY_MARKER
    return [LABEL_KEEP] * n, [[]] * n


def _extract_beat_token_labels(source_beat, target_beat):
    """
    Extract per-token labels for a single beat.

    Returns:
        (labels, targets) - lists matching the source beat's token count
        (including SPLIT/EMPTY marker).
    """
    s_tokens = source_beat['tokens']
    t_tokens = target_beat['tokens']

    # Both empty
    if len(s_tokens) == 0 and len(t_tokens) == 0:
        return [LABEL_KEEP], [[]]  # EMPTY_MARKER → KEEP

    # Source empty, target non-empty → KEEP the EMPTY_MARKER, but APPEND target tokens
    if len(s_tokens) == 0 and len(t_tokens) > 0:
        n_append = min(len(t_tokens), LABEL_APPEND_MAX_N)
        label = label_id_append(n_append)
        return [label], [list(t_tokens[:n_append])]

    # Source non-empty, target empty → DELETE all bundled tokens, KEEP the SPLIT
    if len(s_tokens) > 0 and len(t_tokens) == 0:
        labels = [LABEL_KEEP]  # SPLIT marker → KEEP
        tgts = [[]]
        for _ in s_tokens:
            labels.append(LABEL_DELETE)
            tgts.append([])
        return labels, tgts

    # Both non-empty: identical → all KEEP
    if s_tokens == t_tokens:
        labels = [LABEL_KEEP] * (1 + len(s_tokens))  # SPLIT + tokens
        tgts = [[]] * (1 + len(s_tokens))
        return labels, tgts

    # Both non-empty, different content: pitch-based alignment
    if len(s_tokens) == len(t_tokens):
        return _align_equal_length(s_tokens, t_tokens)
    else:
        return _align_unequal_length(s_tokens, t_tokens)


def _align_equal_length(source_tokens, target_tokens):
    """Token-by-token alignment for equal-length beats."""
    labels = [LABEL_KEEP]  # SPLIT marker
    tgts = [[]]

    for s, t in zip(source_tokens, target_tokens):
        if s == t:
            labels.append(LABEL_KEEP)
            tgts.append([])
        else:
            labels.append(LABEL_REPLACE)
            tgts.append([t])

    return labels, tgts


def _align_unequal_length(source_tokens, target_tokens):
    """
    Pitch-based alignment for unequal-length beats.

    Strategy (same as GECToR with_pair):
    1. Decode both to (abs_pitch, val) lists
    2. Match notes by pitch
    3. Source-only notes → DELETE
    4. Common notes → KEEP or REPLACE
    5. Target-only notes → APPEND on nearest kept source token
    """
    source_notes = decode_beat(source_tokens)
    target_notes = decode_beat(target_tokens)

    source_pitch_to_val = {p: v for p, v in source_notes}
    target_pitch_to_val = {p: v for p, v in target_notes}

    common_pitches = set(source_pitch_to_val) & set(target_pitch_to_val)
    pitches_to_delete = set(source_pitch_to_val) - common_pitches
    pitches_to_insert = sorted(set(target_pitch_to_val) - common_pitches)

    # Build expected bundled tokens for kept notes (using target values)
    kept_notes = sorted([(p, target_pitch_to_val[p]) for p in common_pitches])
    expected_bundled = encode_beat(kept_notes)

    # Generate per-token labels
    labels = [LABEL_KEEP]  # SPLIT marker
    tgts = [[]]

    exp_idx = 0
    for s_note_idx, (s_pitch, s_val) in enumerate(source_notes):
        if s_pitch in pitches_to_delete:
            labels.append(LABEL_DELETE)
            tgts.append([])
        else:
            # Kept note: compare with expected
            if exp_idx < len(expected_bundled):
                expected_token = expected_bundled[exp_idx]
                source_token = source_tokens[s_note_idx]
                if source_token == expected_token:
                    labels.append(LABEL_KEEP)
                    tgts.append([])
                else:
                    labels.append(LABEL_REPLACE)
                    tgts.append([expected_token])
                exp_idx += 1
            else:
                labels.append(LABEL_KEEP)
                tgts.append([])

    # Handle inserts: APPEND target-only notes on the best source token
    if pitches_to_insert:
        _apply_inserts(labels, tgts, source_notes, pitches_to_delete,
                       pitches_to_insert, target_notes, target_pitch_to_val)

    assert len(labels) == 1 + len(source_tokens), \
        f"Labels length {len(labels)} != 1 + source length {len(source_tokens)}"

    return labels, tgts


def _apply_inserts(labels, tgts, source_notes, pitches_to_delete,
                   pitches_to_insert, target_notes, target_pitch_to_val):
    """
    Attach APPEND labels for target-only notes.

    Find the best source token (last kept note with lower pitch) and
    mark it as APPEND_N with the target bundled tokens.
    """
    n_insert = min(len(pitches_to_insert), LABEL_APPEND_MAX_N)
    if n_insert == 0:
        return

    # Build the bundled tokens for the notes to insert
    # We need them in order with correct relative positions
    insert_notes = sorted([(p, target_pitch_to_val[p]) for p in pitches_to_insert[:n_insert]])
    insert_bundled = encode_beat(insert_notes)
    if not insert_bundled:
        return

    first_insert_pitch = pitches_to_insert[0]

    # Find the best label_idx to attach APPEND to:
    # Last kept note with pitch < first_insert_pitch
    best_label_idx = None
    label_idx = 1  # skip SPLIT marker at index 0

    for s_pitch, s_val in source_notes:
        if s_pitch not in pitches_to_delete:
            if s_pitch < first_insert_pitch:
                best_label_idx = label_idx
        label_idx += 1

    if best_label_idx is None:
        best_label_idx = 0  # attach to SPLIT marker

    # Only attach if the position currently has KEEP
    if best_label_idx < len(labels) and labels[best_label_idx] == LABEL_KEEP:
        labels[best_label_idx] = label_id_append(len(insert_bundled))
        tgts[best_label_idx] = list(insert_bundled)
