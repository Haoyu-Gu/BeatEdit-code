"""
Levenshtein Transformer for Music Inpainting.

Encoder-only architecture with 3 prediction heads sharing a BERT backbone:
1. Deletion Head: binary classification per token (KEEP=0 / DELETE=1)
2. Placeholder Head: predict number of placeholders to insert per gap (0..max_insert)
3. Token Head: predict actual token ID for each placeholder position

Architecture: BertModel (shared pre-trained encoder) → 3 Heads

The BERT backbone matches the architecture used in SeqTag and TagFill (§3.1),
enabling proper weight transfer from the shared Music BERT pre-training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from configs.config import LevTModelConfig, PAD_TOKEN, PLH_TOKEN


def _get_bert_config(config: LevTModelConfig) -> BertConfig:
    """Create HuggingFace BertConfig matching Music BERT pre-training."""
    return BertConfig(
        vocab_size=config.vocab_size,  # 7146 (7145 base + PLH)
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        pad_token_id=config.pad_token_id,
        type_vocab_size=1,
    )


class LevenshteinTransformer(nn.Module):
    """
    Levenshtein Transformer for music inpainting.

    Shared BertModel backbone with three task-specific heads.
    """

    def __init__(self, config: LevTModelConfig = None):
        super().__init__()
        if config is None:
            config = LevTModelConfig()
        self.config = config

        # Shared BERT encoder (same architecture as SeqTag/TagFill)
        bert_config = _get_bert_config(config)
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        hidden_size = config.hidden_size

        # Head 1: Deletion classifier (per token, binary)
        self.del_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, 2),  # 0=KEEP, 1=DELETE
        )

        # Head 2: Placeholder insertion predictor (per gap)
        # Input: concatenation of adjacent hidden states → 2 * hidden_size
        self.ins_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 2, config.max_insert + 1),  # 0..max_insert
        )

        # Head 3: Token predictor (per position, full vocabulary)
        self.tok_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.vocab_size),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize task-specific head weights (BERT encoder is init'd by HuggingFace)."""
        for head in [self.del_head, self.ins_head, self.tok_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _encode(self, input_ids, attention_mask, src_mask=None):
        """Run shared BERT encoder backbone.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 1=real, 0=pad
            src_mask: unused, kept for API compatibility
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, H)

    def forward_del(self, input_ids, attention_mask):
        """
        Deletion prediction.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 1=real, 0=pad

        Returns:
            del_logits: (B, L, 2) per-token deletion logits
        """
        hidden = self._encode(input_ids, attention_mask)
        return self.del_head(hidden)

    def forward_ins(self, input_ids, attention_mask):
        """
        Placeholder insertion prediction.

        Predicts how many placeholders to insert between each pair of adjacent tokens.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L)

        Returns:
            ins_logits: (B, L+1, max_insert+1) per-gap insertion count logits
        """
        hidden = self._encode(input_ids, attention_mask)  # (B, L, H)
        B, L, H = hidden.shape

        # Create gap representations by concatenating adjacent hidden states
        # Gap i is between position i-1 and position i
        # For L tokens, there are L+1 gaps (before first, between each pair, after last)

        # Prepend a zero vector for the gap before the first token
        zero_left = torch.zeros(B, 1, H, device=hidden.device)
        zero_right = torch.zeros(B, 1, H, device=hidden.device)

        # hidden_left[i] = hidden state of the token to the left of gap i
        # hidden_right[i] = hidden state of the token to the right of gap i
        hidden_left = torch.cat([zero_left, hidden], dim=1)    # (B, L+1, H)
        hidden_right = torch.cat([hidden, zero_right], dim=1)  # (B, L+1, H)

        gap_hidden = torch.cat([hidden_left, hidden_right], dim=2)  # (B, L+1, 2H)

        return self.ins_head(gap_hidden)  # (B, L+1, max_insert+1)

    def forward_tok(self, input_ids, attention_mask):
        """
        Token prediction (for placeholder positions).

        Args:
            input_ids: (B, L) token IDs (with PLH tokens at positions to fill)
            attention_mask: (B, L)

        Returns:
            tok_logits: (B, L, vocab_size) per-position token prediction logits
        """
        hidden = self._encode(input_ids, attention_mask)
        return self.tok_head(hidden)

    def forward(self, z_ids, attention_mask, operation='all', src_mask=None):
        """
        Combined forward pass.

        Args:
            z_ids: (B, L) intermediate state token IDs
            attention_mask: (B, L)
            operation: 'all' | 'delete' | 'insert' | 'token'
            src_mask: optional attention bias, see _encode()

        Returns:
            dict with 'del_logits', 'ins_logits', 'tok_logits' as applicable
        """
        hidden = self._encode(z_ids, attention_mask, src_mask=src_mask)
        B, L, H = hidden.shape

        result = {}

        if operation in ('all', 'delete'):
            result['del_logits'] = self.del_head(hidden)  # (B, L, 2)

        if operation in ('all', 'insert'):
            zero_left = torch.zeros(B, 1, H, device=hidden.device)
            zero_right = torch.zeros(B, 1, H, device=hidden.device)
            hidden_left = torch.cat([zero_left, hidden], dim=1)
            hidden_right = torch.cat([hidden, zero_right], dim=1)
            gap_hidden = torch.cat([hidden_left, hidden_right], dim=2)
            result['ins_logits'] = self.ins_head(gap_hidden)  # (B, L+1, max_insert+1)

        if operation in ('all', 'token'):
            result['tok_logits'] = self.tok_head(hidden)  # (B, L, vocab_size)

        return result

    def load_pretrained_bert(self, checkpoint_path):
        """
        Load pretrained BERT weights from Music BERT MLM checkpoint.

        Since we now use HuggingFace BertModel as backbone (same as SeqTag),
        weight transfer is direct. Handles vocab size mismatch for the PLH token
        (BERT has 7145 tokens, we have 7146 = 7145 + PLH).
        """
        import os
        from safetensors.torch import load_file

        safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            bin_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location='cpu')
            else:
                # Try loading as a single file checkpoint
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

        # Remove 'module.' prefix if present (DDP)
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            cleaned[new_k] = v

        # Extract BERT encoder weights (strip 'bert.' prefix from BertForMaskedLM keys)
        bert_state = {}
        skipped = []
        for k, v in cleaned.items():
            if k.startswith('bert.'):
                bert_state[k[len('bert.'):]] = v
            else:
                skipped.append(k)

        # Handle vocab size mismatch: BERT has V tokens, we have V+1 (PLH)
        emb_key = 'embeddings.word_embeddings.weight'
        if emb_key in bert_state:
            src_emb = bert_state[emb_key]
            dst_emb = self.bert.embeddings.word_embeddings.weight
            if src_emb.shape[0] < dst_emb.shape[0] and src_emb.shape[1] == dst_emb.shape[1]:
                # Copy all pre-trained embeddings, leave PLH token randomly initialized
                new_emb = dst_emb.data.clone()
                new_emb[:src_emb.shape[0]] = src_emb
                bert_state[emb_key] = new_emb

        # Load into self.bert
        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)

        loaded = sum(1 for k in bert_state if k not in (unexpected or []))
        if missing:
            print(f"[BERT Load] Missing keys (expected for pooler): {missing}")
        if skipped:
            print(f"[BERT Load] Skipped {len(skipped)} non-BERT keys (MLM head etc.)")
        print(f"[BERT Load] Loaded {loaded} weight tensors from {checkpoint_path}")

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
