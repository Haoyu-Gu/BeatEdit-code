"""
Levenshtein alignment utilities for music inpainting.

Ported from fairseq (fairseq/models/nat/levenshtein_utils.py).
Provides:
- Levenshtein distance and edit path computation (DP)
- Insertion label extraction (_get_ins_targets)
- Deletion label extraction (_get_del_targets)
- Sequence manipulation helpers (_apply_del_words, _apply_ins_masks, _apply_ins_words)
"""

import torch
import numpy as np


# ==================== Levenshtein DP (numpy-optimized, per-sample) ====================

def levenshtein_distance_and_path_fast(src, tgt):
    """
    Compute Levenshtein distance and edit path, optimized with numpy.

    Strategy:
    - DP forward pass: Use Python lists for the row-by-row DP (avoids numpy
      scalar indexing overhead). Pre-compute match positions per row using
      numpy vectorized comparison, then use a tight Python inner loop with
      inlined min logic. Store completed rows into a numpy 2D array.
    - Backtrace: Read from the numpy 2D array (O(1) element access via
      integer indexing is fast on contiguous int32 arrays).

    This is ~1.3-1.5x faster than the pure-Python list-of-lists version for
    sequences of length 200-1000+.

    Args:
        src: list of ints (source sequence)
        tgt: list of ints (target sequence)

    Returns:
        distance: int
        path: list of ('keep'|'delete'|'insert'|'replace', src_pos, tgt_pos)
    """
    m, n = len(src), len(tgt)

    # Edge cases
    if m == 0:
        path = [('insert', None, j) for j in range(n)]
        return n, path
    if n == 0:
        path = [('delete', i, None) for i in range(m)]
        return m, path

    src_list = list(src)
    tgt_list = list(tgt)
    tgt_arr = np.array(tgt, dtype=np.int64)

    # DP table stored as numpy for compact memory and fast backtrace
    dp = np.empty((m + 1, n + 1), dtype=np.int32)
    dp[0] = np.arange(n + 1, dtype=np.int32)

    # Previous row as Python list for fast scalar access in inner loop
    prev_row = list(range(n + 1))

    for i in range(1, m + 1):
        src_val = src_list[i - 1]
        # Vectorized match detection: which tgt positions match src[i-1]
        match_flags = (tgt_arr == src_val)  # bool array, shape (n,)
        match_list = match_flags.tolist()   # Python list of bools (fast indexing)

        # Build current row in pure Python (tight inner loop)
        curr_row = [0] * (n + 1)
        curr_row[0] = i
        # Local variable aliases for speed
        pr = prev_row
        cr = curr_row
        ml = match_list

        for j in range(1, n + 1):
            if ml[j - 1]:
                cr[j] = pr[j - 1]
            else:
                # Inlined min of three: pr[j] (delete), cr[j-1] (insert), pr[j-1] (replace)
                a = pr[j]
                b = cr[j - 1]
                c = pr[j - 1]
                if a <= b:
                    cr[j] = 1 + (a if a <= c else c)
                else:
                    cr[j] = 1 + (b if b <= c else c)

        # Store row into numpy array for backtrace
        dp[i] = curr_row
        prev_row = curr_row

    distance = int(prev_row[n])

    # Backtrace using numpy array (contiguous int32, fast element access)
    path = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_list[i - 1] == tgt_list[j - 1]:
            path.append(('keep', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            path.append(('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            path.append(('delete', i - 1, None))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            path.append(('insert', None, j - 1))
            j -= 1
        else:
            break

    path.reverse()
    return distance, path


# ==================== Levenshtein DP (pure Python fallback, per-sample) ====================

def levenshtein_distance_and_path(src, tgt):
    """
    Compute Levenshtein distance and edit path between two token sequences.

    Args:
        src: list of ints (source sequence)
        tgt: list of ints (target sequence)

    Returns:
        distance: int
        path: list of ('keep'|'delete'|'insert'|'replace', src_pos, tgt_pos)
    """
    m, n = len(src), len(tgt)
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # delete
                    dp[i][j - 1],     # insert
                    dp[i - 1][j - 1], # replace
                )

    # Backtrace
    path = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src[i - 1] == tgt[j - 1]:
            path.append(('keep', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            path.append(('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            path.append(('delete', i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            path.append(('insert', None, j - 1))
            j -= 1
        else:
            break

    path.reverse()
    return dp[m][n], path


def compute_edit_labels(z_ids, y_ids):
    """
    Given intermediate state Z and target Y, compute the three label sets
    using Levenshtein alignment.

    Args:
        z_ids: list of int - intermediate/current sequence
        y_ids: list of int - target sequence

    Returns:
        del_labels: list of int, len(z_ids), 0=KEEP 1=DELETE
        ins_labels: list of int, len(z_ids)+1, number of tokens to insert at each gap
        tok_labels: list of int, tokens to fill at placeholder positions
    """
    try:
        dist, path = levenshtein_distance_and_path_fast(z_ids, y_ids)
    except Exception:
        dist, path = levenshtein_distance_and_path(z_ids, y_ids)

    z_len = len(z_ids)
    y_len = len(y_ids)

    # Deletion labels: which z tokens to delete
    del_labels = [0] * z_len  # 0=KEEP

    # Insertion labels: how many tokens to insert at each gap (z_len + 1 gaps)
    ins_labels = [0] * (z_len + 1)

    # Token labels: what tokens to insert
    tok_labels = []

    # Process the edit path
    # We need to track: for each position in z, is it kept or deleted?
    # For each gap in z, how many insertions?

    # Build alignment: map z positions to their fate
    z_pos = 0  # current position in z
    gap_insertions = [[] for _ in range(z_len + 1)]  # insertions before each z position

    for op, s_idx, t_idx in path:
        if op == 'keep':
            # z[s_idx] matches y[t_idx], keep it
            del_labels[s_idx] = 0
            z_pos = s_idx + 1
        elif op == 'replace':
            # z[s_idx] should be deleted, y[t_idx] needs to be inserted
            del_labels[s_idx] = 1
            # After deleting z[s_idx], we need y[t_idx] at this position
            # This maps to: delete z[s_idx], insert y[t_idx] in the gap
            gap_insertions[s_idx].append(y_ids[t_idx])
            z_pos = s_idx + 1
        elif op == 'delete':
            # z[s_idx] should be deleted
            del_labels[s_idx] = 1
            z_pos = s_idx + 1
        elif op == 'insert':
            # y[t_idx] needs to be inserted; find which gap
            # Insert before z_pos (or at end if z_pos == z_len)
            gap_insertions[z_pos].append(y_ids[t_idx])

    # Convert gap_insertions to ins_labels and tok_labels
    for gap_idx in range(z_len + 1):
        ins_labels[gap_idx] = len(gap_insertions[gap_idx])
        tok_labels.extend(gap_insertions[gap_idx])

    return del_labels, ins_labels, tok_labels


def compute_edit_labels_with_context(z_ids, y_ids, mask_start, mask_end):
    """
    Compute edit labels with context constraints for inpainting.

    Only the mask region [mask_start, mask_end) in z_ids is editable.
    Context regions are forced to KEEP with 0 insertions.

    Args:
        z_ids: list of int - intermediate sequence
        y_ids: list of int - target sequence
        mask_start: int - start of mask region in z_ids
        mask_end: int - end of mask region in z_ids (exclusive)

    Returns:
        del_labels, ins_labels, tok_labels (same as compute_edit_labels)
        But with context constraints applied.
    """
    # Extract regions
    z_context_left = z_ids[:mask_start]
    z_mask_region = z_ids[mask_start:mask_end]
    z_context_right = z_ids[mask_end:]

    y_context_left = y_ids[:mask_start]
    y_mask_end_in_y = len(y_ids) - len(z_context_right)
    y_mask_region = y_ids[mask_start:y_mask_end_in_y]
    y_context_right = y_ids[y_mask_end_in_y:]

    # Only align the mask region
    if len(z_mask_region) == 0 and len(y_mask_region) == 0:
        # Nothing to do
        del_labels = [0] * len(z_ids)
        ins_labels = [0] * (len(z_ids) + 1)
        tok_labels = []
        return del_labels, ins_labels, tok_labels

    mask_del, mask_ins, mask_tok = compute_edit_labels(z_mask_region, y_mask_region)

    # Build full labels with context constraints
    z_len = len(z_ids)

    del_labels = [0] * z_len
    ins_labels = [0] * (z_len + 1)

    # Fill mask region labels
    for i, d in enumerate(mask_del):
        del_labels[mask_start + i] = d

    # Insertion labels: mask region gaps are [mask_start, mask_start + len(z_mask_region)]
    # But we need to map them to the full sequence gap indices
    # Gap index i in z corresponds to the gap before z[i]
    # mask_ins[0] = insertions before z_mask_region[0] = gap at mask_start
    # mask_ins[k] = insertions before z_mask_region[k] (or after last) = gap at mask_start+k
    for i, n_ins in enumerate(mask_ins):
        ins_labels[mask_start + i] = n_ins

    tok_labels = mask_tok

    return del_labels, ins_labels, tok_labels


# ==================== Batch Tensor Operations (for training) ====================

def get_ins_targets(src_tokens, tgt_tokens, pad_id, plh_id):
    """
    Batch insertion target computation.

    For each sample in the batch, compute:
    - mask_ins_targets: (B, src_len+1) number of tokens to insert at each gap
    - masked_tgt_tokens: (B, tgt_len) target with inserted positions as PLH
    - masked_tgt_masks: (B, tgt_len) True for positions that were in source

    Args:
        src_tokens: (B, S) source token IDs
        tgt_tokens: (B, T) target token IDs
        pad_id: padding token ID
        plh_id: placeholder token ID

    Returns:
        mask_ins_targets, masked_tgt_tokens, masked_tgt_masks
    """
    B = src_tokens.size(0)
    device = src_tokens.device

    all_ins_targets = []
    all_masked_tgt = []
    all_masked_masks = []

    for b in range(B):
        # Remove padding
        src = src_tokens[b].cpu().tolist()
        tgt = tgt_tokens[b].cpu().tolist()
        src = [t for t in src if t != pad_id]
        tgt = [t for t in tgt if t != pad_id]

        _, ins_labels, tok_labels = compute_edit_labels(src, tgt)

        all_ins_targets.append(ins_labels)

        # Build masked target: source tokens + PLH at insertion points
        masked_tgt = []
        masks = []  # True = from source (kept), False = inserted (PLH)
        tok_idx = 0

        for i in range(len(src) + 1):
            # Insert placeholders before position i
            n_ins = ins_labels[i]
            for _ in range(n_ins):
                if tok_idx < len(tok_labels):
                    masked_tgt.append(tok_labels[tok_idx])
                    tok_idx += 1
                else:
                    masked_tgt.append(plh_id)
                masks.append(False)
            # Add source token (if not past end)
            if i < len(src):
                masked_tgt.append(src[i])
                masks.append(True)

        all_masked_tgt.append(masked_tgt)
        all_masked_masks.append(masks)

    # Pad to same length
    max_ins_len = max(len(x) for x in all_ins_targets)
    max_tgt_len = max(len(x) for x in all_masked_tgt)

    ins_targets_padded = torch.zeros(B, max_ins_len, dtype=torch.long, device=device)
    masked_tgt_padded = torch.full((B, max_tgt_len), pad_id, dtype=torch.long, device=device)
    masked_masks_padded = torch.zeros(B, max_tgt_len, dtype=torch.bool, device=device)

    for b in range(B):
        L = len(all_ins_targets[b])
        ins_targets_padded[b, :L] = torch.tensor(all_ins_targets[b], dtype=torch.long)
        L2 = len(all_masked_tgt[b])
        masked_tgt_padded[b, :L2] = torch.tensor(all_masked_tgt[b], dtype=torch.long)
        masked_masks_padded[b, :L2] = torch.tensor(all_masked_masks[b], dtype=torch.bool)

    return ins_targets_padded, masked_tgt_padded, masked_masks_padded


def get_del_targets(predictions, targets, pad_id):
    """
    Batch deletion target computation.

    For each sample, compute which tokens in predictions should be deleted
    to get closer to targets.

    Args:
        predictions: (B, L) predicted token IDs
        targets: (B, T) target token IDs
        pad_id: padding token ID

    Returns:
        del_targets: (B, L) binary, 1=delete 0=keep
    """
    B = predictions.size(0)
    L = predictions.size(1)
    device = predictions.device

    del_targets = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        pred = predictions[b].cpu().tolist()
        tgt = targets[b].cpu().tolist()
        pred = [t for t in pred if t != pad_id]
        tgt = [t for t in tgt if t != pad_id]

        del_labels, _, _ = compute_edit_labels(pred, tgt)

        for i, d in enumerate(del_labels):
            if i < L:
                del_targets[b, i] = d

    return del_targets


# ==================== Inference Sequence Operations ====================

def apply_del_words(tokens, scores, del_preds, pad_id, bos_id, eos_id):
    """
    Apply deletion predictions to a batch of sequences.

    Args:
        tokens: (B, L) token IDs
        scores: (B, L) token scores/confidences
        del_preds: (B, L) binary predictions, 1=delete
        pad_id, bos_id, eos_id: special token IDs

    Returns:
        new_tokens: (B, L') token IDs after deletion (padded)
        new_scores: (B, L') scores after deletion (padded)
        new_lengths: (B,) actual lengths after deletion
    """
    B, L = tokens.shape
    device = tokens.device

    # Never delete BOS, EOS, PAD
    del_mask = del_preds.clone()
    del_mask.masked_fill_(tokens == bos_id, 0)
    del_mask.masked_fill_(tokens == eos_id, 0)
    del_mask.masked_fill_(tokens == pad_id, 0)

    keep_mask = (1 - del_mask).bool()

    # Count kept tokens per sample
    keep_counts = keep_mask.sum(dim=1)  # (B,)
    max_keep = keep_counts.max().item()

    new_tokens = torch.full((B, max_keep), pad_id, dtype=torch.long, device=device)
    new_scores = torch.zeros(B, max_keep, device=device)

    for b in range(B):
        kept = tokens[b][keep_mask[b]]
        kept_scores = scores[b][keep_mask[b]]
        n = kept.size(0)
        new_tokens[b, :n] = kept
        new_scores[b, :n] = kept_scores

    return new_tokens, new_scores, keep_counts


def apply_ins_masks(tokens, scores, ins_preds, pad_id, plh_id, eos_id):
    """
    Apply insertion predictions: insert PLH tokens at predicted gaps.

    Args:
        tokens: (B, L) token IDs
        scores: (B, L) scores
        ins_preds: (B, L-1) or (B, L+1) number of placeholders to insert per gap
        pad_id, plh_id, eos_id: special token IDs

    Returns:
        new_tokens: (B, L') with PLH inserted
        new_scores: (B, L')
        new_lengths: (B,)
    """
    B, L = tokens.shape
    device = tokens.device

    # Clamp insertion counts
    ins_preds = ins_preds.clamp(min=0, max=255)

    # Calculate new lengths
    # ins_preds[b, i] = number of PLH to insert after tokens[b, i]
    # Ensure ins_preds has right shape
    if ins_preds.size(1) == L - 1:
        # Pad to L (no insertion after last token)
        ins_preds = torch.cat([
            ins_preds,
            torch.zeros(B, 1, dtype=ins_preds.dtype, device=device),
        ], dim=1)

    # Don't insert after PAD tokens
    pad_mask = (tokens == pad_id)
    ins_preds = ins_preds.masked_fill(pad_mask, 0)

    total_ins = ins_preds.sum(dim=1)  # (B,)
    # actual token count (non-pad) per sample
    actual_lens = (tokens != pad_id).sum(dim=1)  # (B,)
    new_lens = actual_lens + total_ins  # (B,)
    max_new_len = new_lens.max().item()

    new_tokens = torch.full((B, max_new_len), pad_id, dtype=torch.long, device=device)
    new_scores = torch.zeros(B, max_new_len, device=device)

    for b in range(B):
        out_idx = 0
        for i in range(L):
            if tokens[b, i].item() == pad_id:
                break
            # Place original token
            new_tokens[b, out_idx] = tokens[b, i]
            new_scores[b, out_idx] = scores[b, i]
            out_idx += 1
            # Insert PLH after this token
            n_ins = ins_preds[b, i].item()
            for _ in range(n_ins):
                if out_idx < max_new_len:
                    new_tokens[b, out_idx] = plh_id
                    new_scores[b, out_idx] = 0.0
                    out_idx += 1

    return new_tokens, new_scores, new_lens


def apply_ins_words(tokens, scores, word_preds, word_scores, plh_id):
    """
    Replace PLH tokens with predicted words.

    Args:
        tokens: (B, L) with PLH tokens
        scores: (B, L) scores
        word_preds: (B, L) predicted token for each position
        word_scores: (B, L) prediction scores
        plh_id: placeholder token ID

    Returns:
        new_tokens: (B, L) with PLH replaced
        new_scores: (B, L) updated scores
    """
    plh_mask = (tokens == plh_id)
    new_tokens = tokens.clone()
    new_scores = scores.clone()

    new_tokens[plh_mask] = word_preds[plh_mask]
    new_scores[plh_mask] = word_scores[plh_mask]

    return new_tokens, new_scores
