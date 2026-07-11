"""
Label extraction for GECToR training data (no_pair encoding).

Aligns source (perturbed) and target (clean) token sequences beat-by-beat,
extracting per-token edit labels from the 350-label space.

Key differences from no_pair_related (Scheme B):
- Absolute positions instead of relative positions
- TRACK markers (TRACK0_START/TRACK1_START) at beat start instead of EMPTY/END
- Beat tokens: [pos][val][pos][val]... (no END_MARKER)
- Empty beat tokens: [0] (single zero, not [EMPTY_MARKER])
- SHIFT labels apply to absolute position tokens

Two alignment strategies:
1. Equal-length beats: direct token comparison (pitch shift, rhythm change)
2. Unequal-length beats: note-level pitch matching (deletion, insertion)
"""

from config import (
    LABEL_KEEP, LABEL_DELETE,
    label_id_keep, label_id_delete, label_id_replace,
    label_id_append, label_id_shift, decode_label,
    is_position_token, is_patch_value, is_control_token, is_music_token,
    POSITION_OFFSET, POSITION_MIN, POSITION_MAX,
    MUSIC_TOKEN_MAX,
)
from sequence_parser import parse_sequence, decode_beat, encode_beat, parse_note_pairs


def extract_labels(source_tokens, target_tokens):
    """
    Extract per-token edit labels by aligning source and target beat-by-beat.

    Since perturbation only modifies beat content (not structure), the number
    and order of beats is preserved. This allows beat-level alignment.

    In Scheme A, each beat in the sequence is:
    - [TRACK_MARKER] + beat_content_tokens
    So labels for each beat include: KEEP for TRACK_MARKER + labels for content.

    Args:
        source_tokens: perturbed token sequence (list of ints)
        target_tokens: clean token sequence (list of ints)

    Returns:
        labels: list of label IDs, same length as source_tokens
    """
    source_info = parse_sequence(source_tokens)
    target_info = parse_sequence(target_tokens)

    labels = []

    # Header tokens -> all KEEP
    for _ in source_info['header_tokens']:
        labels.append(LABEL_KEEP)

    # Process bars and beats
    s_beat_idx = 0
    for bar in source_info['bars']:
        # BAR token -> KEEP
        labels.append(LABEL_KEEP)

        for beat in bar['beats']:
            if s_beat_idx < len(target_info['beats']):
                t_beat = target_info['beats'][s_beat_idx]
                beat_labels = extract_beat_labels(beat, t_beat)
            else:
                beat_labels = _keep_labels_for_beat(beat)

            labels.extend(beat_labels)
            s_beat_idx += 1

    # Footer tokens -> all KEEP
    for _ in source_info['footer_tokens']:
        labels.append(LABEL_KEEP)

    assert len(labels) == len(source_tokens), \
        f"Label length {len(labels)} != source length {len(source_tokens)}"

    return labels


def _keep_labels_for_beat(beat):
    """Generate all-KEEP labels for a beat (TRACK_MARKER + content tokens)."""
    # TRACK_MARKER + content tokens
    return [LABEL_KEEP] * (1 + len(beat['tokens']))


def extract_beat_labels(source_beat, target_beat):
    """
    Extract labels for a single beat by comparing source and target.

    source_beat/target_beat: dict from parse_sequence with:
        'tokens': list of content tokens (without track marker)
        'track_id': TRACK0_START or TRACK1_START

    Returns:
        Label list matching the source beat's full representation in the sequence
        (including TRACK_MARKER).
    """
    s_tokens = source_beat['tokens']
    t_tokens = target_beat['tokens']

    # Both empty: [TRACK_MARKER][0] => KEEP for both
    s_empty = (len(s_tokens) == 1 and s_tokens[0] == 0) or len(s_tokens) == 0
    t_empty = (len(t_tokens) == 1 and t_tokens[0] == 0) or len(t_tokens) == 0

    if s_empty and t_empty:
        # TRACK_MARKER + [0] (or empty content)
        return [LABEL_KEEP] * (1 + len(s_tokens))

    # Source empty, target non-empty: can't edit the 0 token into notes
    if s_empty and not t_empty:
        # Post-processing + iteration needed
        return [LABEL_KEEP] * (1 + len(s_tokens))

    # Source non-empty, target empty: delete all note tokens
    if not s_empty and t_empty:
        labels = [LABEL_KEEP]  # TRACK_MARKER -> KEEP
        for tok in s_tokens:
            if is_music_token(tok):
                labels.append(label_id_delete())
            else:
                labels.append(LABEL_KEEP)
        return labels

    # Both non-empty: compare content
    if s_tokens == t_tokens:
        # Identical
        labels = [LABEL_KEEP]  # TRACK_MARKER
        labels.extend([LABEL_KEEP] * len(s_tokens))
        return labels

    # Equal length: most common case (pitch shift, rhythm change)
    if len(s_tokens) == len(t_tokens):
        return _align_equal_length(s_tokens, t_tokens)

    # Unequal length: note-level matching (deletion or insertion)
    return _align_unequal_length(s_tokens, t_tokens)


# ==================== Equal-Length Alignment ====================

def _align_equal_length(source_beat, target_beat):
    """
    Token-by-token alignment for equal-length beats.

    Position tokens use SHIFT if delta is in [-5, +5], otherwise REPLACE.
    Patch values use REPLACE.
    """
    labels = [LABEL_KEEP]  # TRACK_MARKER -> KEEP

    for s, t in zip(source_beat, target_beat):
        if s == t:
            labels.append(LABEL_KEEP)
        elif is_position_token(s) and is_position_token(t):
            diff = t - s
            if -5 <= diff <= 5 and diff != 0:
                labels.append(label_id_shift(diff))
            else:
                labels.append(label_id_replace(t))
        elif is_patch_value(s) and is_patch_value(t):
            labels.append(label_id_replace(t))
        else:
            # Mixed types or control tokens (shouldn't happen in well-formed data)
            labels.append(label_id_replace(t) if is_music_token(t) else LABEL_KEEP)

    return labels


# ==================== Unequal-Length Alignment ====================

def _align_unequal_length(source_beat, target_beat):
    """
    Note-level pitch matching for unequal-length beats.

    Strategy:
    1. Decode both to (abs_pitch, val) lists
    2. Match notes by pitch
    3. Source-only notes -> DELETE (both position and value tokens)
    4. Common notes -> KEEP/SHIFT/REPLACE position and value tokens
    5. Target-only notes -> APPEND first one (rest deferred to later iterations)

    Note: In Scheme A (absolute positions), after deleting a note the remaining
    positions are still valid (unlike Scheme B where relative positions cascade).
    However, the expected tokens for kept notes may still differ if the target
    was re-encoded with different ordering.
    """
    # Decode both beats to absolute pitch notes
    source_notes = decode_beat(source_beat)
    target_notes = decode_beat(target_beat)

    # Build pitch maps
    source_pitch_to_val = {p: v for p, v in source_notes}
    target_pitch_to_val = {p: v for p, v in target_notes}

    common_pitches = set(source_pitch_to_val) & set(target_pitch_to_val)
    pitches_to_delete = set(source_pitch_to_val) - common_pitches
    pitches_to_insert = sorted(set(target_pitch_to_val) - common_pitches)

    # Build "expected" tokens for kept notes (using target values)
    kept_notes = sorted([(p, target_pitch_to_val[p]) for p in common_pitches])
    if len(kept_notes) > 0:
        expected_tokens = encode_beat(kept_notes)
        expected_pairs = parse_note_pairs(expected_tokens)
    else:
        expected_pairs = []

    # Parse source tokens into note pairs
    source_pairs = parse_note_pairs(source_beat)

    # Generate labels
    labels = [LABEL_KEEP]  # TRACK_MARKER -> KEEP

    exp_idx = 0
    for src_pos, src_val, src_pitch in source_pairs:
        if src_pitch in pitches_to_delete:
            # DELETE both position and value tokens
            labels.append(label_id_delete())
            labels.append(label_id_delete())
        else:
            # Kept note: compare with expected tokens
            if exp_idx < len(expected_pairs):
                exp_pos, exp_val, exp_pitch = expected_pairs[exp_idx]
                assert exp_pitch == src_pitch, \
                    f"Pitch mismatch: expected {exp_pitch}, got {src_pitch}"

                # Position token label
                if src_pos == exp_pos:
                    labels.append(LABEL_KEEP)
                else:
                    diff = exp_pos - src_pos
                    if -5 <= diff <= 5 and diff != 0:
                        labels.append(label_id_shift(diff))
                    else:
                        labels.append(label_id_replace(exp_pos))

                # Value token label
                if src_val == exp_val:
                    labels.append(LABEL_KEEP)
                else:
                    labels.append(label_id_replace(exp_val))

                exp_idx += 1
            else:
                # No more expected notes (shouldn't happen)
                labels.append(LABEL_KEEP)
                labels.append(LABEL_KEEP)

    # Handle inserts: APPEND first target-only note's position token
    if pitches_to_insert and len(labels) > 0:
        _apply_first_insert(
            labels, source_pairs, pitches_to_delete, common_pitches,
            pitches_to_insert, target_beat, target_pitch_to_val
        )

    assert len(labels) == 1 + len(source_beat), \
        f"Labels length {len(labels)} != 1 + source length {len(source_beat)}"

    return labels


def _apply_first_insert(labels, source_pairs, pitches_to_delete, common_pitches,
                         pitches_to_insert, target_beat, target_pitch_to_val):
    """
    Attach the first APPEND label for a target-only note.

    Finds the nearest kept source note with lower pitch and APPENDs
    the position token of the first insert note.

    In Scheme A (absolute positions), the position token for the inserted
    note is simply POSITION_OFFSET + pitch.
    """
    first_insert_pitch = pitches_to_insert[0]

    # The position token for this pitch in Scheme A is direct: POSITION_OFFSET + pitch
    insert_pos_token = POSITION_OFFSET + first_insert_pitch

    if insert_pos_token > MUSIC_TOKEN_MAX:
        return

    # Find the best source token to attach APPEND to:
    # the value token of the highest-pitch kept note below insert_pitch
    best_label_idx = None
    label_idx = 1  # skip TRACK_MARKER at index 0

    for src_pos, src_val, src_pitch in source_pairs:
        if src_pitch in pitches_to_delete:
            label_idx += 2  # 2 DELETE labels
        else:
            if src_pitch < first_insert_pitch:
                best_label_idx = label_idx + 1  # value token index
            label_idx += 2

    if best_label_idx is not None and best_label_idx < len(labels):
        if labels[best_label_idx] == LABEL_KEEP:
            labels[best_label_idx] = label_id_append(insert_pos_token)


# ==================== Levenshtein Alignment (fallback) ====================

def levenshtein_align(source, target):
    """
    Standard Levenshtein DP alignment.

    Returns list of (op, s_idx, t_idx) tuples:
        ('keep', i, j)    - source[i] == target[j]
        ('replace', i, j) - source[i] -> target[j]
        ('delete', i, None) - delete source[i]
        ('insert', None, j) - insert target[j]
    """
    m, n = len(source), len(target)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1],  # replace
                )

    # Backtrack
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
            alignment.append(('keep', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append(('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append(('delete', i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(('insert', None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment
