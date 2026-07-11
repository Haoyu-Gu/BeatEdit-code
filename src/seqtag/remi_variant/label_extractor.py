"""
Label extraction for REMI GECToR training data.

Aligns source (perturbed) and target (clean) REMI token sequences bar-by-bar,
extracting per-token edit labels from the 456-label space.

Key challenge: REMI notes share Position tokens, so insert/delete operations
affect the token structure more than in BEAT encoding. We use bar-level alignment
(bars are preserved by perturbation) with Levenshtein alignment within each bar.
"""

from config import (
    LABEL_KEEP, LABEL_DELETE,
    label_id_keep, label_id_delete, label_id_replace,
    label_id_append, label_id_shift, decode_label,
    is_pitch_token, is_velocity_token, is_duration_token,
    is_position_token, is_music_token, is_special_token,
    BAR_TOKEN, EOS_TOKEN, PAD_TOKEN, BOS_TOKEN,
    MUSIC_TOKEN_MAX,
)
from sequence_parser import parse_sequence, get_bar_token_ranges


def extract_labels(source_tokens, target_tokens):
    """
    Extract per-token edit labels by aligning source and target bar-by-bar.

    Since perturbation only modifies note content within bars (not bar structure),
    the number and order of bars is preserved.

    Args:
        source_tokens: perturbed token sequence (list of ints)
        target_tokens: clean token sequence (list of ints)

    Returns:
        labels: list of label IDs, same length as source_tokens
    """
    source_bars = get_bar_token_ranges(source_tokens)
    target_bars = get_bar_token_ranges(target_tokens)

    labels = [LABEL_KEEP] * len(source_tokens)

    # Find header end (tokens before first bar)
    first_bar_idx = source_bars[0][0] if source_bars else len(source_tokens)
    # Header tokens get KEEP (already set)

    # Align bars
    for bar_i in range(min(len(source_bars), len(target_bars))):
        s_start, s_end = source_bars[bar_i]
        t_start, t_end = target_bars[bar_i]

        # BAR token -> KEEP (already set)
        # Content tokens: s_start+1 to s_end
        s_content = source_tokens[s_start + 1:s_end]
        t_content = target_tokens[t_start + 1:t_end]

        if s_content == t_content:
            # Identical bar content -> all KEEP
            continue

        # Align bar content
        bar_labels = _align_bar_content(s_content, t_content)
        for j, label in enumerate(bar_labels):
            labels[s_start + 1 + j] = label

    assert len(labels) == len(source_tokens), \
        f"Label length {len(labels)} != source length {len(source_tokens)}"

    return labels


def _align_bar_content(source_content, target_content):
    """
    Align source and target bar content tokens using Levenshtein alignment.

    Returns label list for source_content tokens.
    """
    if len(source_content) == 0:
        return []

    if len(target_content) == 0:
        # Target is empty -> delete all music tokens
        return [label_id_delete() if is_music_token(t) else LABEL_KEEP
                for t in source_content]

    if len(source_content) == len(target_content):
        # Same length -> try direct token comparison first
        labels = _align_equal_length(source_content, target_content)
        if labels is not None:
            return labels

    # Use Levenshtein alignment
    return _align_levenshtein(source_content, target_content)


def _align_equal_length(source, target):
    """
    Direct token-by-token alignment for equal-length bar content.

    Position tokens use SHIFT if delta is in [-5, +5].
    Other music tokens use REPLACE.

    Returns labels list, or None if structure mismatch detected.
    """
    labels = []
    for s, t in zip(source, target):
        if s == t:
            labels.append(LABEL_KEEP)
        elif is_position_token(s) and is_position_token(t):
            diff = t - s
            if -5 <= diff <= 5 and diff != 0:
                labels.append(label_id_shift(diff))
            elif is_music_token(t):
                labels.append(label_id_replace(t))
            else:
                labels.append(LABEL_KEEP)
        elif is_music_token(s) and is_music_token(t):
            labels.append(label_id_replace(t))
        elif is_music_token(s) and not is_music_token(t):
            # Structural mismatch
            return None
        else:
            labels.append(LABEL_KEEP)
    return labels


def _align_levenshtein(source, target):
    """
    Levenshtein-based alignment for unequal-length bar content.

    Converts alignment operations to GECToR labels:
    - keep -> LABEL_KEEP
    - replace -> LABEL_REPLACE(target_token)
    - delete -> LABEL_DELETE
    - insert -> LABEL_APPEND (first insert only, attached to preceding kept/replaced token)
    """
    alignment = levenshtein_align(source, target)

    # Build labels
    labels = [LABEL_KEEP] * len(source)
    pending_insert = None  # token to append

    for op, s_idx, t_idx in alignment:
        if op == 'keep':
            if pending_insert is not None and s_idx is not None:
                # Attach pending insert to this position
                insert_tok = pending_insert
                if is_music_token(insert_tok):
                    labels[s_idx] = label_id_append(insert_tok)
                pending_insert = None
            # else: already KEEP
        elif op == 'replace':
            t_tok = target[t_idx]
            s_tok = source[s_idx]
            if is_position_token(s_tok) and is_position_token(t_tok):
                diff = t_tok - s_tok
                if -5 <= diff <= 5 and diff != 0:
                    labels[s_idx] = label_id_shift(diff)
                elif is_music_token(t_tok):
                    labels[s_idx] = label_id_replace(t_tok)
            elif is_music_token(t_tok):
                labels[s_idx] = label_id_replace(t_tok)
            if pending_insert is not None:
                pending_insert = None  # drop if can't attach
        elif op == 'delete':
            if is_music_token(source[s_idx]):
                labels[s_idx] = label_id_delete()
            if pending_insert is not None:
                pending_insert = None  # drop
        elif op == 'insert':
            t_tok = target[t_idx]
            if is_music_token(t_tok) and pending_insert is None:
                pending_insert = t_tok

    # Try to attach remaining pending insert to last non-deleted source token
    if pending_insert is not None:
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] == LABEL_KEEP and is_music_token(source[i]) and \
               is_music_token(pending_insert):
                labels[i] = label_id_append(pending_insert)
                break

    return labels


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
