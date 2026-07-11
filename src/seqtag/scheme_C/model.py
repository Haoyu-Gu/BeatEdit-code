"""
MusicGECToR Model (with_pair encoding)

GECToR-style sequence tagging model for music error correction.
Uses a pretrained Music BERT encoder with two classification heads:
- error_detector: binary (correct/error) per token
- tag_predictor: 14258-class tag prediction per token
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM

from config import (
    NUM_LABELS, VOCAB_SIZE, PAD_TOKEN,
    TRAINING_DEFAULTS,
)


def get_bert_config():
    """Create BERT config matching the pretrained Music BERT (with_pair)."""
    return BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=PAD_TOKEN,
        type_vocab_size=1,
    )


class MusicGECToR(nn.Module):
    """
    GECToR-style model for music sequence correction (with_pair).

    Architecture:
        BERT encoder → hidden states → error_detector (2-class)
                                     → tag_predictor (14258-class)
    """

    def __init__(self, num_labels=NUM_LABELS, dropout=0.1, bert_config=None):
        super().__init__()

        if bert_config is None:
            bert_config = get_bert_config()

        self.bert = BertModel(bert_config, add_pooling_layer=False)
        hidden_size = bert_config.hidden_size

        self.error_detector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

        self.tag_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=real, 0=padding
            token_type_ids: (batch, seq_len) segment IDs (unused, all 0)

        Returns:
            detect_logits: (batch, seq_len, 2)
            tag_logits: (batch, seq_len, num_labels)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state

        detect_logits = self.error_detector(hidden_states)
        tag_logits = self.tag_predictor(hidden_states)

        return detect_logits, tag_logits

    def freeze_bert(self):
        """Freeze BERT encoder parameters (Stage I cold start)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """Unfreeze BERT encoder parameters (Stage I fine-tune)."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_head_parameters(self):
        """Get parameters of classification heads only."""
        return list(self.error_detector.parameters()) + \
               list(self.tag_predictor.parameters())

    def get_bert_parameters(self):
        """Get BERT encoder parameters."""
        return list(self.bert.parameters())


def load_pretrained_bert(checkpoint_path, model=None):
    """
    Load pretrained BERT weights from an Accelerate checkpoint.

    The Music BERT was trained as BertForMaskedLM and saved via
    accelerator.save_state(). The checkpoint contains model.safetensors
    with keys like 'bert.embeddings.word_embeddings.weight'.

    Args:
        checkpoint_path: path to the checkpoint directory (containing model.safetensors)
        model: optional MusicGECToR model to load weights into.
               If None, creates a new one.

    Returns:
        MusicGECToR model with pretrained BERT weights.
    """
    if model is None:
        model = MusicGECToR()

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
            raise FileNotFoundError(
                f"No model weights found in {checkpoint_path}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )

    # The checkpoint has keys like 'bert.encoder.layer.0...'
    # Our model also has 'bert.encoder.layer.0...'
    # Plus it has 'cls.predictions...' for MLM head (which we ignore)
    bert_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            bert_state[k[len('bert.'):]] = v  # strip 'bert.' prefix
        else:
            skipped.append(k)

    # Load into model.bert
    missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)

    if missing:
        print(f"[BERT Load] Missing keys: {missing}")
    if unexpected:
        print(f"[BERT Load] Unexpected keys: {unexpected}")
    if skipped:
        print(f"[BERT Load] Skipped {len(skipped)} non-BERT keys (MLM head etc.)")

    return model


def compute_loss(detect_logits, tag_logits, detect_labels, tag_labels,
                 attention_mask=None, keep_weight=0.15, lambda_detect=0.5):
    """
    Compute GECToR dual-head loss.

    Args:
        detect_logits: (batch, seq_len, 2)
        tag_logits: (batch, seq_len, num_labels)
        detect_labels: (batch, seq_len) values 0/1/-100
        tag_labels: (batch, seq_len) values 0-14257/-100
        attention_mask: (batch, seq_len) optional
        keep_weight: class weight for KEEP label (label 0)
        lambda_detect: weight for detection loss

    Returns:
        (total_loss, tag_loss, detect_loss)
    """
    num_labels = tag_logits.shape[-1]

    # KEEP label downweighting
    tag_weights = torch.ones(num_labels, device=tag_logits.device)
    tag_weights[0] = keep_weight

    # Tag prediction loss (main)
    tag_loss_fn = nn.CrossEntropyLoss(weight=tag_weights, ignore_index=-100)
    tag_loss = tag_loss_fn(
        tag_logits.view(-1, num_labels),
        tag_labels.view(-1),
    )

    # Error detection loss (auxiliary)
    detect_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    detect_loss = detect_loss_fn(
        detect_logits.view(-1, 2),
        detect_labels.view(-1),
    )

    total_loss = tag_loss + lambda_detect * detect_loss
    return total_loss, tag_loss, detect_loss
