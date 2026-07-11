"""
FELIX-Music Token-Level Label Extractor (Scheme A: no_pair).

Aligns source (perturbed) and target (clean) token sequences beat-by-beat,
extracting per-token edit labels from the 11-label space.

Like GECToR: one label per source token.
  KEEP=0, DELETE=1, REPLACE=2, APPEND_1..8 = 3..10

Key differences from Scheme C (with_pair):
- No SPLIT markers; beats use TRACK markers (TRACK0_START, TRACK1_START)
- Empty beats: [TRACK_MARKER][0] (track marker + single 0 token)
- Non-empty beats: [TRACK_MARKER][pos][val][pos][val]... (absolute positions)
- Each note = 2 tokens: deletion/replacement operates on token pairs
"""

from configs.config import (
    LABEL_KEEP, LABEL_DELETE, LABEL_REPLACE,
    label_id_append, LABEL_APPEND_MAX_N,
    POSITION_OFFSET,
    is_position_token, is_patch_value, is_track_marker,
)
from data.sequence_parser import parse_sequence, decode_beat, encode_beat


def _is_empty_beat_tokens(tokens):
    """Check if beat tokens represent an empty beat."""
    return len(tokens) == 0 or (len(tokens) == 1 and tokens[0] == 0)


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

    # Header tokens -> all KEEP
    for _ in source_info['header_tokens']:
        labels.append(LABEL_KEEP)
        targets.append([])

    # Process bars and beats
    s_beat_idx = 0
    for bar in source_info['bars']:
        # BAR token -> KEEP
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

    # Footer tokens -> all KEEP
    for _ in source_info['footer_tokens']:
        labels.append(LABEL_KEEP)
        targets.append([])

    assert len(labels) == len(source_tokens), \
        f"Label length {len(labels)} != source length {len(source_tokens)}"
    assert len(targets) == len(source_tokens)

    return labels, targets


def _keep_labels_for_beat(beat):
    """Generate all-KEEP labels for a beat (track marker + content tokens)."""
    # Track marker + content tokens
    n = 1 + len(beat['tokens'])
    return [LABEL_KEEP] * n, [[]] * n


def _extract_beat_token_labels(source_beat, target_beat):
    """
    Extract per-token labels for a single beat.

    In Scheme A, beat tokens in the sequence are:
    - [TRACK_MARKER][content_tokens...]
    - Empty: [TRACK_MARKER][0]  (2 tokens)
    - Non-empty: [TRACK_MARKER][pos][val][pos][val]...  (1 + 2*N tokens)

    Returns:
        (labels, targets) - lists matching the source beat's token count
        (track marker + content tokens).
    """
    s_tokens = source_beat['tokens']
    t_tokens = target_beat['tokens']

    s_empty = _is_empty_beat_tokens(s_tokens)
    t_empty = _is_empty_beat_tokens(t_tokens)

    # Both empty
    if s_empty and t_empty:
        # TRACK_MARKER -> KEEP, [0] -> KEEP
        return [LABEL_KEEP, LABEL_KEEP], [[], []]

    # Source empty, target non-empty -> APPEND target tokens onto the [0] token
    if s_empty and not t_empty:
        n_append = min(len(t_tokens), LABEL_APPEND_MAX_N)
        label = label_id_append(n_append)
        # TRACK_MARKER -> KEEP, [0] -> APPEND_N
        return [LABEL_KEEP, label], [[], list(t_tokens[:n_append])]

    # Source non-empty, target empty -> KEEP track marker, DELETE all content tokens
    if not s_empty and t_empty:
        labels = [LABEL_KEEP]  # TRACK_MARKER
        tgts = [[]]
        for _ in s_tokens:
            labels.append(LABEL_DELETE)
            tgts.append([])
        return labels, tgts

    # Both non-empty: identical -> all KEEP
    if s_tokens == t_tokens:
        labels = [LABEL_KEEP] * (1 + len(s_tokens))  # TRACK + tokens
        tgts = [[]] * (1 + len(s_tokens))
        return labels, tgts

    # Both non-empty, different content: alignment needed
    if len(s_tokens) == len(t_tokens):
        return _align_equal_length(s_tokens, t_tokens)
    else:
        return _align_unequal_length(s_tokens, t_tokens)


def _align_equal_length(source_tokens, target_tokens):
    """Token-by-token alignment for equal-length beats."""
    labels = [LABEL_KEEP]  # TRACK_MARKER
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

    Strategy:
    1. Decode both to (abs_pitch, val) note lists
    2. Match notes by pitch
    3. Source-only notes -> DELETE (both pos and val tokens)
    4. Common notes -> KEEP/REPLACE per token
    5. Target-only notes -> APPEND on nearest kept source token
    """
    source_notes = decode_beat(source_tokens)
    target_notes = decode_beat(target_tokens)

    source_pitch_to_val = {p: v for p, v in source_notes}
    target_pitch_to_val = {p: v for p, v in target_notes}

    common_pitches = set(source_pitch_to_val) & set(target_pitch_to_val)
    pitches_to_delete = set(source_pitch_to_val) - common_pitches
    pitches_to_insert = sorted(set(target_pitch_to_val) - common_pitches)

    # Build expected tokens for kept notes (using target values, re-encoded)
    kept_notes = sorted([(p, target_pitch_to_val[p]) for p in common_pitches])
    expected_tokens = encode_beat(kept_notes)
    # For non-empty, encode_beat returns [pos,val,pos,val,...] (no [0] prefix)
    # For empty kept_notes, it returns [0] which we treat as empty
    if expected_tokens == [0]:
        expected_tokens = []

    # Generate per-token labels
    labels = [LABEL_KEEP]  # TRACK_MARKER
    tgts = [[]]

    exp_idx = 0  # index into expected_tokens (pos-val pairs)
    i = 0
    note_idx = 0

    while i < len(source_tokens):
        tok = source_tokens[i]

        if is_position_token(tok) and i + 1 < len(source_tokens) and is_patch_value(source_tokens[i + 1]):
            # This is a note (pos + val pair)
            if note_idx < len(source_notes):
                s_pitch, s_val = source_notes[note_idx]

                if s_pitch in pitches_to_delete:
                    # DELETE both tokens
                    labels.append(LABEL_DELETE)
                    tgts.append([])
                    labels.append(LABEL_DELETE)
                    tgts.append([])
                else:
                    # Kept note: compare with expected
                    if exp_idx + 1 < len(expected_tokens):
                        exp_pos = expected_tokens[exp_idx]
                        exp_val = expected_tokens[exp_idx + 1]

                        # Position token
                        if tok == exp_pos:
                            labels.append(LABEL_KEEP)
                            tgts.append([])
                        else:
                            labels.append(LABEL_REPLACE)
                            tgts.append([exp_pos])

                        # Value token
                        if source_tokens[i + 1] == exp_val:
                            labels.append(LABEL_KEEP)
                            tgts.append([])
                        else:
                            labels.append(LABEL_REPLACE)
                            tgts.append([exp_val])

                        exp_idx += 2
                    else:
                        # No more expected tokens - keep as is
                        labels.append(LABEL_KEEP)
                        tgts.append([])
                        labels.append(LABEL_KEEP)
                        tgts.append([])

                note_idx += 1
            else:
                # Extra note beyond decoded notes - keep
                labels.append(LABEL_KEEP)
                tgts.append([])
                labels.append(LABEL_KEEP)
                tgts.append([])

            i += 2
        else:
            # Non-note token - keep
            labels.append(LABEL_KEEP)
            tgts.append([])
            i += 1

    # Handle inserts: APPEND target-only notes on the best source token
    if pitches_to_insert:
        _apply_inserts(labels, tgts, source_tokens, source_notes,
                       pitches_to_delete, pitches_to_insert, target_pitch_to_val)

    assert len(labels) == 1 + len(source_tokens), \
        f"Labels length {len(labels)} != 1 + source length {len(source_tokens)}"

    return labels, tgts


def _apply_inserts(labels, tgts, source_tokens, source_notes, pitches_to_delete,
                   pitches_to_insert, target_pitch_to_val):
    """
    Attach APPEND labels for target-only notes.

    For Scheme A, find the best source token (last kept note's value token)
    and mark it as APPEND_N with the target tokens.
    """
    n_insert = min(len(pitches_to_insert), LABEL_APPEND_MAX_N // 2)  # 2 tokens per note
    if n_insert == 0:
        return

    # Build the tokens for the notes to insert
    insert_notes = sorted([(p, target_pitch_to_val[p]) for p in pitches_to_insert[:n_insert]])
    insert_tokens = encode_beat(insert_notes)
    # encode_beat returns [0] for empty, but insert_notes is non-empty here
    if insert_tokens == [0]:
        return
    if not insert_tokens:
        return

    # Cap at LABEL_APPEND_MAX_N
    insert_tokens = insert_tokens[:LABEL_APPEND_MAX_N]

    first_insert_pitch = pitches_to_insert[0]

    # Find the best label_idx to attach APPEND to:
    # Last kept note's value token with pitch < first_insert_pitch
    best_label_idx = None
    label_idx = 1  # skip TRACK_MARKER at index 0
    note_idx = 0

    i = 0
    while i < len(source_tokens):
        tok = source_tokens[i]
        if is_position_token(tok) and i + 1 < len(source_tokens) and is_patch_value(source_tokens[i + 1]):
            if note_idx < len(source_notes):
                s_pitch = source_notes[note_idx][0]
                if s_pitch not in pitches_to_delete and s_pitch < first_insert_pitch:
                    best_label_idx = label_idx + 1  # value token position
                note_idx += 1
            label_idx += 2
            i += 2
        else:
            label_idx += 1
            i += 1

    if best_label_idx is None:
        best_label_idx = 0  # attach to TRACK_MARKER

    # Only attach if the position currently has KEEP
    if best_label_idx < len(labels) and labels[best_label_idx] == LABEL_KEEP:
        labels[best_label_idx] = label_id_append(len(insert_tokens))
        tgts[best_label_idx] = list(insert_tokens)
