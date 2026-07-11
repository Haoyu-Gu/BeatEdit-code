"""
Label extraction for GECToR training data (with_pair encoding).

Aligns source (perturbed) and target (clean) token sequences beat-by-beat,
extracting per-token edit labels from the 14258-label space.

Key advantage over no_pair_related:
- 1 bundled token = 1 complete note → 1 REPLACE = complete note correction
- No SHIFT labels needed (pitch change = REPLACE entire bundled token)
- INSERT needs only 1 APPEND label per note (not 2+)
- DELETE needs only 1 DELETE label per note (not 2)
"""

from config import (
    LABEL_KEEP, LABEL_DELETE,
    label_id_keep, label_id_delete, label_id_replace, label_id_append,
    decode_label, is_bundled_token, is_control_token,
    EMPTY_MARKER, PATTERN_NUM, BUNDLED_TOKEN_MAX,
)
from sequence_parser import parse_sequence, decode_beat, encode_beat


def extract_labels(source_tokens, target_tokens):
    """
    Extract per-token edit labels by aligning source and target beat-by-beat.

    Since perturbation only modifies beat content (not structure), the number
    and order of beats is preserved. This allows beat-level alignment.

    Args:
        source_tokens: perturbed token sequence (list of ints)
        target_tokens: clean token sequence (list of ints)

    Returns:
        labels: list of label IDs, same length as source_tokens
    """
    source_info = parse_sequence(source_tokens)
    target_info = parse_sequence(target_tokens)

    labels = []

    # Header tokens → all KEEP
    for _ in source_info['header_tokens']:
        labels.append(LABEL_KEEP)

    # Process bars and beats
    s_beat_idx = 0
    for bar in source_info['bars']:
        # BAR token → KEEP
        labels.append(LABEL_KEEP)

        for beat in bar['beats']:
            if s_beat_idx < len(target_info['beats']):
                t_beat = target_info['beats'][s_beat_idx]
                beat_labels = extract_beat_labels(beat, t_beat)
            else:
                beat_labels = _keep_labels_for_beat(beat)

            labels.extend(beat_labels)
            s_beat_idx += 1

    # Footer tokens → all KEEP
    for _ in source_info['footer_tokens']:
        labels.append(LABEL_KEEP)

    assert len(labels) == len(source_tokens), \
        f"Label length {len(labels)} != source length {len(source_tokens)}"

    return labels


def _keep_labels_for_beat(beat):
    """Generate all-KEEP labels for a beat (including SPLIT/EMPTY marker)."""
    if beat['split_id'] is not None:
        # SPLIT marker + bundled tokens
        return [LABEL_KEEP] * (1 + len(beat['tokens']))
    else:
        # EMPTY_MARKER
        return [LABEL_KEEP]


def extract_beat_labels(source_beat, target_beat):
    """
    Extract labels for a single beat by comparing source and target.

    source_beat/target_beat: dict from parse_sequence with:
        'tokens': list of bundled tokens (no SPLIT marker)
        'split_id': SPLIT_0 or SPLIT_1 or None

    Returns:
        Label list matching the source beat's full representation in the sequence
        (including SPLIT/EMPTY marker).
    """
    s_tokens = source_beat['tokens']
    t_tokens = target_beat['tokens']
    s_split = source_beat.get('split_id', None)

    # Both empty
    if len(s_tokens) == 0 and len(t_tokens) == 0:
        return [LABEL_KEEP]  # EMPTY_MARKER

    # Source empty, target non-empty: can't edit EMPTY_MARKER directly
    if len(s_tokens) == 0 and len(t_tokens) > 0:
        return [LABEL_KEEP]  # post-processing + iteration needed

    # Source non-empty, target empty: delete all bundled tokens
    if len(s_tokens) > 0 and len(t_tokens) == 0:
        labels = [LABEL_KEEP]  # SPLIT marker → KEEP
        labels.extend([label_id_delete()] * len(s_tokens))
        return labels

    # Both non-empty: compare content
    if s_tokens == t_tokens:
        # Identical
        labels = [LABEL_KEEP]  # SPLIT marker
        labels.extend([LABEL_KEEP] * len(s_tokens))
        return labels

    # Equal length: token-by-token comparison
    if len(s_tokens) == len(t_tokens):
        return _align_equal_length(s_tokens, t_tokens, s_split)

    # Unequal length: note-level pitch matching
    return _align_unequal_length(s_tokens, t_tokens, s_split)


# ==================== Equal-Length Alignment ====================

def _align_equal_length(source_tokens, target_tokens, split_id):
    """
    Token-by-token alignment for equal-length beats.

    Each bundled token is a complete note, so REPLACE handles everything.
    """
    labels = [LABEL_KEEP]  # SPLIT marker → KEEP

    for s, t in zip(source_tokens, target_tokens):
        if s == t:
            labels.append(LABEL_KEEP)
        else:
            # One REPLACE covers the entire note (pos + value bundled)
            labels.append(label_id_replace(t))

    return labels


# ==================== Unequal-Length Alignment ====================

def _align_unequal_length(source_tokens, target_tokens, split_id):
    """
    Note-level pitch matching for unequal-length beats.

    Strategy:
    1. Decode both to (abs_pitch, val) lists
    2. Match notes by pitch
    3. Source-only notes → DELETE (1 label per note!)
    4. Common notes → KEEP or REPLACE bundled token
    5. Target-only notes → APPEND (1 label per note, 1 round!)
    """
    source_notes = decode_beat(source_tokens)
    target_notes = decode_beat(target_tokens)

    # Build pitch maps
    source_pitch_to_val = {p: v for p, v in source_notes}
    target_pitch_to_val = {p: v for p, v in target_notes}

    common_pitches = set(source_pitch_to_val) & set(target_pitch_to_val)
    pitches_to_delete = set(source_pitch_to_val) - common_pitches
    pitches_to_insert = sorted(set(target_pitch_to_val) - common_pitches)

    # Build expected bundled tokens for kept notes
    kept_notes = sorted([(p, target_pitch_to_val[p]) for p in common_pitches])
    expected_bundled = encode_beat(kept_notes)

    # Generate labels
    labels = [LABEL_KEEP]  # SPLIT marker → KEEP

    # Walk through source notes and assign labels
    exp_idx = 0
    for s_note_idx, (s_pitch, s_val) in enumerate(source_notes):
        if s_pitch in pitches_to_delete:
            # DELETE this note (1 label for 1 bundled token)
            labels.append(label_id_delete())
        else:
            # Kept note: compare with expected bundled token
            if exp_idx < len(expected_bundled):
                expected_token = expected_bundled[exp_idx]
                source_token = source_tokens[s_note_idx]
                if source_token == expected_token:
                    labels.append(LABEL_KEEP)
                else:
                    labels.append(label_id_replace(expected_token))
                exp_idx += 1
            else:
                labels.append(LABEL_KEEP)

    # Handle inserts: APPEND first target-only note
    if pitches_to_insert:
        _apply_first_insert(
            labels, source_notes, pitches_to_delete, common_pitches,
            pitches_to_insert, target_notes, target_pitch_to_val
        )

    assert len(labels) == 1 + len(source_tokens), \
        f"Labels length {len(labels)} != 1 + source length {len(source_tokens)}"

    return labels


def _apply_first_insert(labels, source_notes, pitches_to_delete, common_pitches,
                         pitches_to_insert, target_notes, target_pitch_to_val):
    """
    Attach the first APPEND label for a target-only note.

    In with_pair, APPEND adds one complete bundled token = one complete note.
    Find the nearest kept source note with lower pitch and APPEND after it.

    The bundled token must use the correct relative position from the predecessor
    note so that decode_beat() produces the correct absolute pitch.
    """
    first_insert_pitch = pitches_to_insert[0]
    first_insert_val = target_pitch_to_val[first_insert_pitch]

    # Find the best source token to attach APPEND to:
    # The last kept note with pitch < first_insert_pitch
    best_label_idx = None
    best_pitch = 0  # pitch of the predecessor note (default: 0 = beat start)
    label_idx = 1  # skip SPLIT marker at index 0

    for s_pitch, s_val in source_notes:
        if s_pitch in pitches_to_delete:
            label_idx += 1  # 1 DELETE label
        else:
            if s_pitch < first_insert_pitch:
                best_label_idx = label_idx
                best_pitch = s_pitch
            label_idx += 1

    # Compute bundled token with correct relative position from predecessor
    if best_label_idx is not None:
        insert_rel_pos = first_insert_pitch - best_pitch
    else:
        # No kept note with lower pitch; attach to SPLIT marker at index 0
        # Relative position is from 0 (beat start)
        insert_rel_pos = first_insert_pitch
        best_label_idx = 0

    insert_bundled = insert_rel_pos * PATTERN_NUM + first_insert_val

    if insert_bundled > BUNDLED_TOKEN_MAX:
        return  # out of range

    if best_label_idx is not None and best_label_idx < len(labels):
        if labels[best_label_idx] == LABEL_KEEP:
            labels[best_label_idx] = label_id_append(insert_bundled)


# ==================== Levenshtein Alignment (fallback) ====================

def levenshtein_align(source, target):
    """
    Standard Levenshtein DP alignment.

    Returns list of (op, s_idx, t_idx) tuples:
        ('keep', i, j)    - source[i] == target[j]
        ('replace', i, j) - source[i] → target[j]
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
