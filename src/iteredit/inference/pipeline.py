"""
Iterative decoding pipeline for Levenshtein Transformer Music Inpainting.

Implements the three-step iterative refinement:
1. Deletion: predict and remove unwanted tokens
2. Placeholder insertion: predict and insert PLH tokens at gaps
3. Token fill: predict actual tokens for PLH positions

Uses per-token `editable` flags instead of fragile index-based mask
boundary tracking.  Each token carries a boolean indicating whether it
belongs to the editable (inpainted) region.  Context tokens are always
frozen: they cannot be deleted, and no new tokens can be inserted
between two frozen tokens.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F

from configs.config import (
    LevTModelConfig, PAD_TOKEN, PLH_TOKEN, BOS_TOKEN, EOS_TOKEN,
    is_header_token, is_bundled_token,
)
from models.levenshtein_transformer import LevenshteinTransformer
from data.levenshtein_utils import apply_del_words, apply_ins_masks, apply_ins_words


class LevTInpaintingPipeline:
    """
    Iterative decoding pipeline for music inpainting.

    Given a masked sequence (with a gap where beats were removed),
    iteratively refine until the gap is filled.

    Editability is tracked with a per-token boolean list ``editable``
    rather than explicit mask-start / mask-end indices:
    - ``editable[i] = True``  -> token *i* lives in the inpainted region
      and may be deleted or replaced.
    - ``editable[i] = False`` -> token *i* is frozen context.

    Insertion is allowed at gap *i* (the gap **before** token *i*) when
    the gap touches the editable region: either ``editable[i-1]`` or
    ``editable[i]`` is True, or the gap is at the original mask
    boundary.  Newly inserted PLH tokens and subsequently filled tokens
    are always marked editable.
    """

    def __init__(self, checkpoint_path, device='cuda', config=None):
        """
        Args:
            checkpoint_path: path to trained LevT checkpoint
            device: torch device
            config: optional LevTModelConfig override
        """
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        if config is None:
            if 'config' in ckpt:
                config = LevTModelConfig(**ckpt['config'])
            else:
                config = LevTModelConfig()

        self.config = config
        self.model = LevenshteinTransformer(config)

        state_dict = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gap_is_insertable(editable, gap_idx):
        """Return True if gap *gap_idx* borders at least one editable token.

        Gap *gap_idx* sits **before** ``current[gap_idx]`` (with gap 0
        before the first token, and gap ``len(current)`` after the last).

        A gap is insertable when it is adjacent to at least one editable
        token.
        """
        L = len(editable)
        left_editable = (gap_idx > 0) and editable[gap_idx - 1]
        right_editable = (gap_idx < L) and editable[gap_idx]
        return left_editable or right_editable

    def _make_tensors(self, current):
        """Build (1, L) input_ids and attention_mask tensors."""
        seq_tensor = torch.tensor(
            [current], dtype=torch.long, device=self.device
        )
        attn_mask = torch.ones(
            1, len(current), dtype=torch.long, device=self.device
        )
        return seq_tensor, attn_mask

    # ------------------------------------------------------------------
    # Main inpainting loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        masked_seq,
        mask_start,
        mask_end=None,
        max_iter=10,
        temperature=1.0,
        del_threshold=0.5,
        top_k=0,
        verbose=False,
    ):
        """
        Perform iterative inpainting on a masked sequence.

        Args:
            masked_seq: list of int, token sequence with the masked region
                        REMOVED (context_left + context_right, no gap).
            mask_start: int, position in *masked_seq* where the gap is.
                        After removal the left context occupies indices
                        ``[0, mask_start)`` and right context starts at
                        ``mask_start``.
            mask_end:   int, same as mask_start for a clean removal
                        (if None, defaults to mask_start).
            max_iter:   max number of refinement iterations.
            temperature: sampling temperature for token prediction
                         (<=0 or 1.0 -> greedy argmax).
            del_threshold: confidence threshold for deletion head.
            top_k:      if >0, restrict token sampling to top-k logits.
            verbose:    print iteration details.

        Returns:
            completed_seq: list of int, the inpainted sequence.
            history:       list of intermediate sequences.
        """
        if mask_end is None:
            mask_end = mask_start

        # ---- Initialise current sequence and editable flags ----
        # All tokens in masked_seq are context (the masked region was
        # removed), so every token starts as non-editable.
        current = list(masked_seq)
        editable = [False] * len(current)

        # The ``insertion_seed`` marks the gap at mask_start as the
        # initial insertion point so that the first iteration can inject
        # PLH tokens even though no editable tokens exist yet.
        insertion_seed = mask_start

        history = [list(current)]

        for step in range(max_iter):
            seq_tensor, attn_mask = self._make_tensors(current)

            # Track per-step stats for convergence detection.
            n_deleted = 0
            total_insertions = 0
            n_filled = 0

            # ---- Step 1: Deletion ----
            # Only attempt deletion from step 1 onward (step 0 has no
            # editable tokens yet) and only when there are editable
            # tokens to consider.
            n_editable = sum(editable)
            if step > 0 and n_editable > 0:
                output = self.model(seq_tensor, attn_mask, operation='delete')
                del_logits = output['del_logits']          # (1, L, 2)
                del_probs = F.softmax(del_logits, dim=-1)
                del_preds = (del_probs[:, :, 1] > del_threshold).long()  # (1, L)

                # Context constraint: freeze non-editable tokens.
                for i in range(len(current)):
                    if not editable[i]:
                        del_preds[0, i] = 0
                # Never delete BOS / EOS / PAD regardless of editable.
                for i, tok in enumerate(current):
                    if tok in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                        del_preds[0, i] = 0

                n_deleted = del_preds[0].sum().item()

                if n_deleted > 0:
                    keep_mask = (del_preds[0] == 0)
                    new_current = []
                    new_editable = []
                    for i in range(len(current)):
                        if keep_mask[i]:
                            new_current.append(current[i])
                            new_editable.append(editable[i])
                    current = new_current
                    editable = new_editable

                    if verbose:
                        print(
                            f"  Step {step}: deleted {n_deleted} tokens, "
                            f"seq_len={len(current)}"
                        )

                    # Rebuild tensors after deletion.
                    seq_tensor, attn_mask = self._make_tensors(current)

            # ---- Step 2: Placeholder Insertion ----
            output = self.model(seq_tensor, attn_mask, operation='insert')
            ins_logits = output['ins_logits']      # (1, L+1, max_insert+1)
            ins_preds = ins_logits.argmax(dim=-1)  # (1, L+1)

            L = len(current)

            # Context constraint: zero out insertions at gaps that are
            # not adjacent to any editable token.
            for gap in range(L + 1):
                if insertion_seed is not None and gap == insertion_seed:
                    # Always allow the seed gap on the very first step.
                    continue
                if not self._gap_is_insertable(editable, gap):
                    ins_preds[0, gap] = 0

            total_insertions = ins_preds[0].sum().item()

            if total_insertions > 0:
                new_current = []
                new_editable = []

                for i in range(L + 1):
                    # Insert PLH tokens before position i.
                    n_ins = ins_preds[0, i].item()
                    for _ in range(n_ins):
                        new_current.append(PLH_TOKEN)
                        new_editable.append(True)  # new PLH is editable

                    if i < L:
                        new_current.append(current[i])
                        new_editable.append(editable[i])

                current = new_current
                editable = new_editable

                if verbose:
                    print(
                        f"  Step {step}: inserted {total_insertions} "
                        f"placeholders, seq_len={len(current)}"
                    )

                # Rebuild tensors after insertion.
                seq_tensor, attn_mask = self._make_tensors(current)

            # After the first insertion round, disable the seed -- from
            # now on insertability is governed purely by editable flags.
            if total_insertions > 0:
                insertion_seed = None

            # ---- Step 3: Token Fill ----
            plh_positions = [
                i for i, t in enumerate(current) if t == PLH_TOKEN
            ]
            n_filled = len(plh_positions)

            if n_filled > 0:
                output = self.model(seq_tensor, attn_mask, operation='token')
                tok_logits = output['tok_logits']  # (1, L, vocab_size)

                for pos in plh_positions:
                    logits = tok_logits[0, pos]    # (vocab_size,)
                    predicted = self._sample_token(
                        logits, temperature=temperature, top_k=top_k
                    )
                    current[pos] = predicted
                    # Filled token remains editable.
                    editable[pos] = True

                if verbose:
                    print(
                        f"  Step {step}: filled {n_filled} placeholders"
                    )

            history.append(list(current))

            # ---- Step 4: Convergence Check ----
            if n_deleted == 0 and total_insertions == 0 and n_filled == 0:
                if verbose:
                    print(f"  Converged at step {step}")
                break

            # Also stop when no PLH tokens remain AND nothing was
            # deleted or inserted (steady state).
            if PLH_TOKEN not in current:
                if n_deleted == 0 and total_insertions == 0:
                    if verbose:
                        print(
                            f"  All placeholders filled at step {step}"
                        )
                    break

        return current, history

    # ------------------------------------------------------------------
    # Token sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(logits, temperature=1.0, top_k=0):
        """Sample a single token from *logits* (1-D).

        Args:
            logits:      (vocab_size,) raw logits tensor.
            temperature: softmax temperature.  <=0 or ==1.0 -> greedy.
            top_k:       if >0, keep only the top-k logits before
                         sampling.

        Returns:
            int: predicted token ID.
        """
        if temperature <= 0 or temperature == 1.0:
            if top_k <= 0:
                return logits.argmax().item()
            # top-k greedy: still just argmax
            return logits.argmax().item()

        scaled = logits / temperature

        if top_k > 0:
            top_vals, top_idx = torch.topk(scaled, min(top_k, scaled.size(0)))
            # Build a full-size tensor of -inf, scatter top-k back.
            filtered = torch.full_like(scaled, float('-inf'))
            filtered.scatter_(0, top_idx, top_vals)
            scaled = filtered

        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1).item()

    # ------------------------------------------------------------------
    # High-level convenience: mask beats from a full sequence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint_from_full(
        self,
        full_seq,
        mask_beat_start,
        mask_beat_end,
        max_iter=10,
        temperature=1.0,
        top_k=0,
        verbose=False,
    ):
        """
        Inpaint by masking specific beats from a full sequence.

        Args:
            full_seq:        list of int, complete token sequence.
            mask_beat_start: first beat index to mask.
            mask_beat_end:   last beat index to mask (exclusive).
            max_iter, temperature, top_k, verbose: forwarded to inpaint().

        Returns:
            completed_seq, history
        """
        from data.sequence_parser import parse_sequence

        parsed = parse_sequence(full_seq)
        beats = parsed['beats']

        if mask_beat_end > len(beats):
            mask_beat_end = len(beats)

        tok_start = beats[mask_beat_start]['start_idx']
        if mask_beat_end < len(beats):
            tok_end = beats[mask_beat_end]['start_idx']
        else:
            if parsed['footer_tokens']:
                tok_end = len(full_seq) - len(parsed['footer_tokens'])
            else:
                tok_end = len(full_seq)

        # Remove the masked region
        masked_seq = full_seq[:tok_start] + full_seq[tok_end:]

        return self.inpaint(
            masked_seq,
            mask_start=tok_start,
            mask_end=tok_start,  # gap = single point after removal
            max_iter=max_iter,
            temperature=temperature,
            top_k=top_k,
            verbose=verbose,
        )


def main():
    """Quick test / demo."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_npz', type=str, required=True)
    parser.add_argument('--mask_beat_start', type=int, default=4)
    parser.add_argument('--mask_beat_end', type=int, default=12)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load and tokenize input
    from data.tokenizer import create_tokenizer
    from data.dataset import LevTDataset
    import numpy as np

    pipeline = LevTInpaintingPipeline(args.checkpoint, device=args.device)

    # Simple demo: create a dummy dataset to tokenize one file
    ds = LevTDataset([args.input_npz], max_len=2048)
    full_tokens = ds._tokenize_npz(0)

    print(f"Full sequence length: {len(full_tokens)} tokens")

    completed, history = pipeline.inpaint_from_full(
        full_tokens,
        mask_beat_start=args.mask_beat_start,
        mask_beat_end=args.mask_beat_end,
        max_iter=args.max_iter,
        temperature=args.temperature,
        top_k=args.top_k,
        verbose=True,
    )

    print(f"Completed sequence length: {len(completed)} tokens")
    print(f"Iterations: {len(history) - 1}")


if __name__ == '__main__':
    main()
