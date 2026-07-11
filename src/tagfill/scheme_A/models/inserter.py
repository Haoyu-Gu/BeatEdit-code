"""
FELIX-Music Inserter Model.

MLM-style model that fills MASK tokens in skeleton sequences.
Uses shared BERT backbone for proper pre-training weight transfer (§3.1).

Input: skeleton sequence with MASK tokens at positions to fill
Output: predicted token IDs for each MASK position

Architecture:
  BertModel (pre-trained, shared backbone)
  → MLM Head (Linear → GELU → LayerNorm → Linear → vocab_size)
  Applied at MASK positions only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from configs.config import InserterConfig, PAD_TOKEN, MASK_TOKEN, VOCAB_SIZE


def _get_bert_config(config: InserterConfig) -> BertConfig:
    """Create HuggingFace BertConfig matching Music BERT pre-training."""
    return BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        pad_token_id=PAD_TOKEN,
        type_vocab_size=1,
    )


class FELIXInserter(nn.Module):
    """
    FELIX Inserter: MLM-style model for filling MASK tokens.

    Args:
        config: InserterConfig with model hyperparameters
    """

    def __init__(self, config: InserterConfig = None):
        super().__init__()
        if config is None:
            config = InserterConfig()
        self.config = config

        # Shared BERT encoder backbone
        bert_config = _get_bert_config(config)
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize MLM head weights."""
        for module in self.mlm_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, skeleton_ids, attention_mask, mask_positions):
        """
        Forward pass.

        Args:
            skeleton_ids: (B, L) token IDs with MASK_TOKEN at positions to fill
            attention_mask: (B, L) 1=real, 0=padding
            mask_positions: (B, M) indices into skeleton_ids for MASK positions
                           -1 entries are padding (ignored)

        Returns:
            logits: (B, M, vocab_size) predictions for each MASK position
        """
        # Encode
        outputs = self.bert(
            input_ids=skeleton_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)

        # Gather representations at mask positions
        mask_reps = self._gather_mask_reps(hidden, mask_positions)  # (B, M, H)

        # Predict
        logits = self.mlm_head(mask_reps)  # (B, M, vocab_size)

        return logits

    def _gather_mask_reps(self, hidden, mask_positions):
        """
        Gather hidden representations at MASK positions.

        Args:
            hidden: (B, L, H)
            mask_positions: (B, M) with -1 for padding

        Returns:
            mask_reps: (B, M, H)
        """
        B, L, H = hidden.shape
        M = mask_positions.shape[1]

        # Clamp -1 to 0 for gathering (will be zeroed out later)
        safe_positions = mask_positions.clamp(min=0)  # (B, M)
        indices = safe_positions.unsqueeze(-1).expand(B, M, H)  # (B, M, H)
        mask_reps = torch.gather(hidden, dim=1, index=indices)  # (B, M, H)

        # Zero out padding positions
        valid_mask = (mask_positions >= 0).unsqueeze(-1).float()  # (B, M, 1)
        mask_reps = mask_reps * valid_mask

        return mask_reps

    def load_pretrained_bert(self, checkpoint_path):
        """
        Load pretrained BERT weights from Music BERT MLM checkpoint.

        Direct weight transfer via HuggingFace BertModel (same architecture).
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
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

        # Remove 'module.' prefix (DDP)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace('module.', '')] = v

        # Extract BERT weights (strip 'bert.' prefix from BertForMaskedLM)
        bert_state = {}
        skipped = []
        for k, v in cleaned.items():
            if k.startswith('bert.'):
                bert_state[k[len('bert.'):]] = v
            else:
                skipped.append(k)

        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)

        loaded = sum(1 for k in bert_state if k not in (unexpected or []))
        if missing:
            print(f"[BERT Load] Missing keys: {missing}")
        if skipped:
            print(f"[BERT Load] Skipped {len(skipped)} non-BERT keys (MLM head etc.)")
        print(f"[BERT Load] Loaded {loaded} weight tensors into Inserter from {checkpoint_path}")
