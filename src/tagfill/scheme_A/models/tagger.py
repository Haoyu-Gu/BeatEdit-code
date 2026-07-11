"""
FELIX-Music Tagger Model.

Token-level sequence labeling using shared BERT backbone:
  BertModel (pre-trained) → Linear → per-token logits (11 classes)

Uses HuggingFace BertModel for proper weight transfer from Music BERT
pre-training (§3.1), consistent with SeqTag and IterEdit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from configs.config import TaggerConfig, PAD_TOKEN, NUM_FELIX_LABELS


def _get_bert_config(config: TaggerConfig) -> BertConfig:
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


class FELIXTagger(nn.Module):
    """
    FELIX Tagger: per-token sequence labeling model.

    Architecture: BertModel (shared pre-trained encoder)
                  → Linear classifier per token
    """

    def __init__(self, config: TaggerConfig = None):
        super().__init__()
        if config is None:
            config = TaggerConfig()
        self.config = config

        # Shared BERT encoder backbone
        bert_config = _get_bert_config(config)
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        # Per-token classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize classification head weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 1=real, 0=padding

        Returns:
            logits: (B, L, num_labels) per-token classification logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(hidden)    # (B, L, num_labels)
        return logits

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
        print(f"[BERT Load] Loaded {loaded} weight tensors into Tagger from {checkpoint_path}")

    def freeze_bert(self):
        """Freeze BERT encoder parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """Unfreeze BERT encoder parameters."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_head_parameters(self):
        """Get classification head parameters."""
        return list(self.classifier.parameters())

    def get_bert_parameters(self):
        """Get BERT encoder parameters."""
        return list(self.bert.parameters())


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, L, C) or (B*L, C)
            targets: (B, L) or (B*L,)
        """
        if logits.dim() == 3:
            B, L, C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(-1)

        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=logits.device)[valid_targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
