"""
Label extraction for CPWord GECToR training data.

Aligns source (perturbed) and target (clean) CPWord compound token sequences bar-by-bar,
extracting per-token factored edit labels.

Simpler than REMI because each note is 1 compound token:
- KEEP: all sub-tokens identical
- REPLACE: any sub-token differs
- DELETE/INSERT: handled via Levenshtein alignment on compound tokens
"""

from config import (
    ACTION_KEEP, ACTION_DELETE, ACTION_REPLACE, ACTION_APPEND,
    LABEL_PAD,
    is_bar_token, is_note_token, is_position_token,
    is_special_token, is_eos_token, is_music_token,
)
from sequence_parser import get_bar_ranges


def extract_labels(source_tokens, target_tokens):
    """
    Extract per-token factored edit labels by aligning source and target bar-by-bar.

    Args:
        source_tokens: perturbed compound token sequence (list of [5])
        target_tokens: clean compound token sequence (list of [5])

    Returns:
        action_labels: list of action IDs (0-3), same length as source_tokens
        sub_token_labels: list of [5] sub-token targets, -100 for KEEP/DELETE positions
    """
    source_bars = get_bar_ranges(source_tokens)
    target_bars = get_bar_ranges(target_tokens)

    action_labels = [ACTION_KEEP] * len(source_tokens)
    sub_token_labels = [[-100] * 5 for _ in range(len(source_tokens))]

    for bar_i in range(min(len(source_bars), len(target_bars))):
        s_start, s_end = source_bars[bar_i]
        t_start, t_end = target_bars[bar_i]

        # Bar token -> KEEP (already set)
        # Content tokens: s_start+1 to s_end
        s_content = source_tokens[s_start + 1:s_end]
        t_content = target_tokens[t_start + 1:t_end]

        if _tokens_equal(s_content, t_content):
            continue

        bar_action_labels, bar_sub_labels = _align_bar_content(s_content, t_content)
        for j, (act, sub) in enumerate(zip(bar_action_labels, bar_sub_labels)):
            action_labels[s_start + 1 + j] = act
            sub_token_labels[s_start + 1 + j] = sub

    return action_labels, sub_token_labels


def _tokens_equal(a, b):
    """Check if two compound token lists are identical."""
    if len(a) != len(b):
        return False
    return all(ta == tb for ta, tb in zip(a, b))


def _align_bar_content(source_content, target_content):
    """
    Align source and target bar content (compound tokens after the Bar token).

    Returns:
        action_labels: list of action IDs for source_content
        sub_token_labels: list of [5] sub-token targets
    """
    if len(source_content) == 0:
        return [], []

    if len(target_content) == 0:
        actions = []
        subs = []
        for tok in source_content:
            if is_music_token(tok):
                actions.append(ACTION_DELETE)
            else:
                actions.append(ACTION_KEEP)
            subs.append([-100] * 5)
        return actions, subs

    if len(source_content) == len(target_content):
        result = _align_equal_length(source_content, target_content)
        if result is not None:
            return result

    return _align_levenshtein(source_content, target_content)


def _align_equal_length(source, target):
    """
    Direct token-by-token alignment for equal-length bar content.

    Returns (action_labels, sub_token_labels) or None if structure mismatch.
    """
    actions = []
    subs = []

    for s_tok, t_tok in zip(source, target):
        if s_tok == t_tok:
            actions.append(ACTION_KEEP)
            subs.append([-100] * 5)
        elif _same_token_type(s_tok, t_tok):
            # Same type but different content -> REPLACE
            actions.append(ACTION_REPLACE)
            subs.append(list(t_tok))
        else:
            # Structure mismatch -> fall back to Levenshtein
            return None

    return actions, subs


def _same_token_type(a, b):
    """Check if two compound tokens are the same type (both notes, both positions, etc.)."""
    if is_note_token(a) and is_note_token(b):
        return True
    if is_position_token(a) and is_position_token(b):
        return True
    if is_bar_token(a) and is_bar_token(b):
        return True
    return False


def _align_levenshtein(source, target):
    """
    Levenshtein-based alignment for unequal-length bar content.

    Returns (action_labels, sub_token_labels) for source tokens.
    """
    alignment = levenshtein_align(source, target)

    actions = [ACTION_KEEP] * len(source)
    subs = [[-100] * 5 for _ in range(len(source))]
    pending_insert = None  # compound token to append

    for op, s_idx, t_idx in alignment:
        if op == 'keep':
            if pending_insert is not None and s_idx is not None:
                if is_music_token(pending_insert):
                    actions[s_idx] = ACTION_APPEND
                    subs[s_idx] = list(pending_insert)
                pending_insert = None
        elif op == 'replace':
            t_tok = target[t_idx]
            if is_music_token(t_tok):
                actions[s_idx] = ACTION_REPLACE
                subs[s_idx] = list(t_tok)
            if pending_insert is not None:
                pending_insert = None
        elif op == 'delete':
            if is_music_token(source[s_idx]):
                actions[s_idx] = ACTION_DELETE
            if pending_insert is not None:
                pending_insert = None
        elif op == 'insert':
            t_tok = target[t_idx]
            if is_music_token(t_tok) and pending_insert is None:
                pending_insert = t_tok

    # Try to attach remaining pending insert to last non-deleted source token
    if pending_insert is not None:
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == ACTION_KEEP and is_music_token(source[i]) and \
               is_music_token(pending_insert):
                actions[i] = ACTION_APPEND
                subs[i] = list(pending_insert)
                break

    return actions, subs


def _compound_eq(a, b):
    """Check if two compound tokens are equal."""
    return a == b


def _compound_match_key(tok):
    """Key for matching compound tokens in Levenshtein alignment.
    Notes match by pitch; position/bar tokens match by full content."""
    if is_note_token(tok):
        return ('note', tok[2])  # match by pitch sub-token
    return ('other', tuple(tok))


def levenshtein_align(source, target):
    """
    Levenshtein DP alignment for compound tokens.

    Uses pitch as the primary matching key for note tokens.

    Returns list of (op, s_idx, t_idx) tuples.
    """
    m, n = len(source), len(target)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _compound_eq(source[i - 1], target[j - 1]):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Give lower cost to replacing same-type tokens
                replace_cost = 1
                if _compound_match_key(source[i - 1]) == _compound_match_key(target[j - 1]):
                    replace_cost = 0  # same pitch note -> "keep" with sub-token diff
                dp[i][j] = min(
                    dp[i - 1][j] + 1,           # delete
                    dp[i][j - 1] + 1,            # insert
                    dp[i - 1][j - 1] + replace_cost,  # replace
                )

    # Backtrack
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and _compound_eq(source[i - 1], target[j - 1]):
            alignment.append(('keep', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0:
            replace_cost = 1
            if _compound_match_key(source[i - 1]) == _compound_match_key(target[j - 1]):
                replace_cost = 0
            if dp[i][j] == dp[i - 1][j - 1] + replace_cost:
                if replace_cost == 0:
                    alignment.append(('keep', i - 1, j - 1))
                else:
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
