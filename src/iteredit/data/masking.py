"""
Beat-level masking strategies for music inpainting.

Provides various strategies to create (masked_seq, full_seq, mask_start, mask_end)
pairs for training the Levenshtein Transformer.
"""

import random
from data.sequence_parser import parse_sequence, get_beat_boundaries
from configs.config import (
    BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    EMPTY_MARKER, SPLIT_0, SPLIT_1,
    is_header_token, is_bundled_token,
)


def create_inpainting_pair(token_ids, mask_ratio_min=0.125, mask_ratio_max=0.5):
    """
    Create an inpainting pair by masking consecutive beats.

    Args:
        token_ids: list of int, complete BEAT token sequence
        mask_ratio_min: minimum fraction of beats to mask
        mask_ratio_max: maximum fraction of beats to mask

    Returns:
        dict with:
            'masked_ids': token sequence with masked region removed
            'full_ids': original complete sequence
            'mask_start': token index where mask starts in full_ids
            'mask_end': token index where mask ends in full_ids (exclusive)
            'masked_region_start': where the gap is in masked_ids
    """
    parsed = parse_sequence(token_ids)
    beats = parsed['beats']
    num_beats = len(beats)

    if num_beats < 4:
        # Too few beats to mask
        return None

    # Determine mask length (in beats)
    min_beats = max(2, int(num_beats * mask_ratio_min))
    max_beats = max(min_beats, int(num_beats * mask_ratio_max))
    mask_len = random.randint(min_beats, max_beats)
    mask_len = min(mask_len, num_beats - 2)  # keep at least 2 beats as context

    # Random start position (leave at least 1 beat on each side)
    max_start = num_beats - mask_len
    if max_start <= 0:
        mask_beat_start = 0
    else:
        mask_beat_start = random.randint(
            min(1, max_start),
            max(1, max_start - 1)
        )
    mask_beat_end = mask_beat_start + mask_len

    # Convert beat indices to token indices
    tok_start = beats[mask_beat_start]['start_idx']
    if mask_beat_end < num_beats:
        tok_end = beats[mask_beat_end]['start_idx']
    else:
        # Mask to end of content (before EOS)
        if parsed['footer_tokens']:
            tok_end = len(token_ids) - len(parsed['footer_tokens'])
        else:
            tok_end = len(token_ids)

    # Create masked sequence (remove the masked region)
    masked_ids = token_ids[:tok_start] + token_ids[tok_end:]

    return {
        'masked_ids': masked_ids,
        'full_ids': list(token_ids),
        'mask_start': tok_start,
        'mask_end': tok_end,
        'masked_region_start': tok_start,  # where gap is in masked_ids
        'num_masked_beats': mask_len,
        'num_total_beats': num_beats,
    }


def create_scattered_mask(token_ids, mask_ratio=0.3):
    """
    Create a mask by randomly removing scattered beats (non-contiguous).

    Args:
        token_ids: list of int
        mask_ratio: fraction of beats to mask

    Returns:
        Same dict format as create_inpainting_pair, but mask_start/mask_end
        may not be contiguous. Returns list of masked beat indices instead.
    """
    parsed = parse_sequence(token_ids)
    beats = parsed['beats']
    num_beats = len(beats)

    if num_beats < 4:
        return None

    # Select random beats to mask
    n_mask = max(1, int(num_beats * mask_ratio))
    # Don't mask first or last beat
    maskable = list(range(1, num_beats - 1))
    if len(maskable) < n_mask:
        n_mask = len(maskable)
    masked_beat_indices = sorted(random.sample(maskable, n_mask))

    # Collect token ranges to remove
    remove_ranges = []
    for bi in masked_beat_indices:
        beat = beats[bi]
        remove_ranges.append((beat['start_idx'], beat['end_idx']))

    # Build masked sequence by removing these ranges
    remove_set = set()
    for start, end in remove_ranges:
        for i in range(start, end):
            remove_set.add(i)

    masked_ids = [t for i, t in enumerate(token_ids) if i not in remove_set]

    return {
        'masked_ids': masked_ids,
        'full_ids': list(token_ids),
        'masked_beat_indices': masked_beat_indices,
        'remove_ranges': remove_ranges,
        'num_masked_beats': n_mask,
        'num_total_beats': num_beats,
    }


def sample_intermediate_state(target_ids, mask_start, mask_end, vocab_size,
                              delete_prob=0.3, replace_prob=0.2):
    """
    Sample an intermediate state by corrupting the target's mask region.

    Simulates what the sequence might look like during iterative refinement:
    some tokens deleted, some replaced with random tokens, some kept correct.

    Args:
        target_ids: list of int, complete target sequence
        mask_start: int, start of mask region
        mask_end: int, end of mask region (exclusive)
        vocab_size: int, vocabulary size for random replacement
        delete_prob: probability of deleting a token
        replace_prob: probability of replacing with random token

    Returns:
        intermediate: list of int, corrupted sequence
        new_mask_start: int, start of mask region in intermediate
        new_mask_end: int, end of mask region in intermediate
    """
    context_left = target_ids[:mask_start]
    context_right = target_ids[mask_end:]
    mask_region = target_ids[mask_start:mask_end]

    corrupted = []
    for tok in mask_region:
        r = random.random()
        if r < delete_prob:
            continue  # delete
        elif r < delete_prob + replace_prob:
            corrupted.append(random.randint(0, vocab_size - 1))  # random replace
        else:
            corrupted.append(tok)  # keep correct

    intermediate = context_left + corrupted + context_right
    new_mask_start = len(context_left)
    new_mask_end = new_mask_start + len(corrupted)

    return intermediate, new_mask_start, new_mask_end
