"""
FELIX-Music Skeleton Builder (Token-Level).

Constructs Inserter input from token-level Tagger labels.
Walks through source tokens and labels to produce a skeleton with MASKs.

Label handling:
  KEEP         → copy token to skeleton
  DELETE       → skip (omit from skeleton)
  REPLACE      → put MASK_TOKEN in skeleton (target = correct token)
  APPEND_N     → copy token + N MASK_TOKENs after (targets = N correct tokens)
"""

from configs.config import (
    LABEL_KEEP, LABEL_DELETE, LABEL_REPLACE,
    LABEL_APPEND_OFFSET, LABEL_APPEND_MAX_N,
    MASK_TOKEN, PAD_TOKEN,
    decode_felix_label,
)


def build_skeleton(source_tokens, labels, targets):
    """
    Build the Inserter's input skeleton from token-level Tagger labels.

    Args:
        source_tokens: list of int, source token sequence
        labels: list of int, per-token FELIX label IDs
        targets: list of lists, per-token target tokens for MASKs

    Returns:
        dict with:
            'skeleton_tokens': list of ints (sequence with MASKs)
            'mask_positions': list of int positions in skeleton that are MASK
            'mask_targets': list of int token IDs for each MASK position
    """
    assert len(source_tokens) == len(labels) == len(targets)

    skeleton = []
    mask_positions = []
    mask_targets = []

    for i, (tok, label, tgt_list) in enumerate(zip(source_tokens, labels, targets)):
        op, value = decode_felix_label(label)

        if op == 'KEEP':
            skeleton.append(tok)

        elif op == 'DELETE':
            pass  # skip this token

        elif op == 'REPLACE':
            skeleton.append(MASK_TOKEN)
            mask_positions.append(len(skeleton) - 1)
            if tgt_list:
                mask_targets.append(tgt_list[0])
            else:
                mask_targets.append(PAD_TOKEN)

        elif op == 'APPEND':
            n = value
            # Keep this token
            skeleton.append(tok)
            # Append N MASKs after
            for j in range(n):
                skeleton.append(MASK_TOKEN)
                mask_positions.append(len(skeleton) - 1)
                if j < len(tgt_list):
                    mask_targets.append(tgt_list[j])
                else:
                    mask_targets.append(PAD_TOKEN)

    return {
        'skeleton_tokens': skeleton,
        'mask_positions': mask_positions,
        'mask_targets': mask_targets,
    }
