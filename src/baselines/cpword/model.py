"""
CP Music GECToR Model

GECToR-style sequence tagging model for music error correction with CPWord encoding.
Uses sum-embedding for compound tokens + BertEncoder + factored prediction heads.

Architecture:
    CompoundEmbedding (5 tables, summed -> 512-dim)
        -> BertEncoder (8L/512H/8A)
        -> error_detector (2-class)
        -> action_head (4-class: KEEP/DELETE/REPLACE/APPEND)
        -> 5 sub-token heads (family/position/pitch/velocity/duration)
"""

import os
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from config import (
    SUB_VOCAB_SIZES, NUM_SUBVOCABS,
    NUM_ACTIONS, PAD_ID, MASK_ID, IGNORE_ID,
    ACTION_KEEP, ACTION_REPLACE, ACTION_APPEND,
    LABEL_PAD, TRAINING_DEFAULTS as TD,
)


class CompoundEmbedding(nn.Module):
    """
    Sum-embedding for CPWord compound tokens.

    Each compound token has 5 sub-tokens. Each sub-token gets its own embedding table.
    The 5 embeddings are summed, then LayerNorm'd.
    """

    def __init__(self, sub_vocab_sizes=SUB_VOCAB_SIZES, hidden_size=512):
        super().__init__()
        self.num_subvocabs = len(sub_vocab_sizes)
        self.hidden_size = hidden_size
        self.sub_vocab_sizes = sub_vocab_sizes

        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, hidden_size, padding_idx=PAD_ID)
            for vs in sub_vocab_sizes
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, compound_ids):
        """
        Args:
            compound_ids: (batch, seq_len, 5) tensor of sub-token IDs

        Returns:
            (batch, seq_len, hidden_size) summed embeddings
        """
        emb_sum = None
        for i, emb_table in enumerate(self.embeddings):
            sub_ids = compound_ids[:, :, i]  # (batch, seq_len)
            emb = emb_table(sub_ids)         # (batch, seq_len, hidden)
            if emb_sum is None:
                emb_sum = emb
            else:
                emb_sum = emb_sum + emb
        return self.layer_norm(emb_sum)


def get_bert_config(hidden_size=512):
    """Create BertConfig for CPWord encoding (no word_embeddings needed)."""
    return BertConfig(
        vocab_size=2,  # dummy, we replace embeddings
        hidden_size=hidden_size,
        num_hidden_layers=TD['num_hidden_layers'],
        num_attention_heads=TD['num_attention_heads'],
        intermediate_size=TD['intermediate_size'],
        max_position_embeddings=TD['max_position_embeddings'],
        hidden_dropout_prob=TD['dropout'],
        attention_probs_dropout_prob=TD['dropout'],
        pad_token_id=0,
        type_vocab_size=1,
    )


class CPBertForMaskedLM(nn.Module):
    """
    BERT MLM with CompoundEmbedding for CPWord pretraining.

    Uses 5 separate MLM heads (one per sub-vocabulary) instead of one flat head.
    """

    def __init__(self, sub_vocab_sizes=SUB_VOCAB_SIZES, hidden_size=512):
        super().__init__()
        config = get_bert_config(hidden_size)
        self.compound_embedding = CompoundEmbedding(sub_vocab_sizes, hidden_size)

        self.bert = BertModel(config, add_pooling_layer=False)
        # Remove BERT's word embeddings (we use CompoundEmbedding)
        self.bert.embeddings.word_embeddings = nn.Identity()

        self.mlm_heads = nn.ModuleList([
            nn.Linear(hidden_size, vs)
            for vs in sub_vocab_sizes
        ])

        self.hidden_size = hidden_size

    def forward(self, compound_ids, attention_mask=None, mlm_labels=None):
        """
        Args:
            compound_ids: (batch, seq_len, 5)
            attention_mask: (batch, seq_len)
            mlm_labels: (batch, seq_len, 5) with -100 for non-masked

        Returns:
            If mlm_labels given: (loss, logits_list)
            Else: logits_list  (list of 5 tensors, each (batch, seq_len, sub_vocab_size))
        """
        embedded = self.compound_embedding(compound_ids)

        # Feed through BERT encoder (bypass word_embeddings via inputs_embeds)
        outputs = self.bert(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        logits_list = [head(hidden_states) for head in self.mlm_heads]

        if mlm_labels is not None:
            loss = torch.tensor(0.0, device=hidden_states.device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            for i, logits in enumerate(logits_list):
                sub_labels = mlm_labels[:, :, i]  # (batch, seq_len)
                loss = loss + loss_fn(
                    logits.view(-1, logits.size(-1)),
                    sub_labels.view(-1),
                )
            return loss, logits_list

        return logits_list

    def get_encoder_state_dict(self):
        """Get state dict of compound_embedding + bert for loading into GECToR."""
        state = {}
        for k, v in self.compound_embedding.state_dict().items():
            state[f'compound_embedding.{k}'] = v
        for k, v in self.bert.state_dict().items():
            state[f'bert.{k}'] = v
        return state


class CPMusicGECToR(nn.Module):
    """
    GECToR-style model for music sequence correction with CPWord encoding.

    Architecture:
        CompoundEmbedding -> BertEncoder -> error_detector (2)
                                         -> action_head (4)
                                         -> family_head (6)
                                         -> position_head (38)
                                         -> pitch_head (156)
                                         -> velocity_head (37)
                                         -> duration_head (69)
    """

    def __init__(self, sub_vocab_sizes=SUB_VOCAB_SIZES, hidden_size=512,
                 dropout=0.1):
        super().__init__()
        config = get_bert_config(hidden_size)
        self.compound_embedding = CompoundEmbedding(sub_vocab_sizes, hidden_size)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = nn.Identity()

        self.error_detector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

        self.action_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_ACTIONS),
        )

        # Sub-token prediction heads (for REPLACE/APPEND)
        self.sub_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vs),
            )
            for vs in sub_vocab_sizes
        ])

        self.hidden_size = hidden_size
        self.sub_vocab_sizes = sub_vocab_sizes

    def forward(self, compound_ids, attention_mask=None):
        """
        Args:
            compound_ids: (batch, seq_len, 5)
            attention_mask: (batch, seq_len)

        Returns:
            detect_logits: (batch, seq_len, 2)
            action_logits: (batch, seq_len, 4)
            sub_logits: list of 5 tensors, each (batch, seq_len, sub_vocab_size)
        """
        embedded = self.compound_embedding(compound_ids)

        outputs = self.bert(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        detect_logits = self.error_detector(hidden_states)
        action_logits = self.action_head(hidden_states)
        sub_logits = [head(hidden_states) for head in self.sub_heads]

        return detect_logits, action_logits, sub_logits

    def freeze_bert(self):
        for param in self.compound_embedding.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.compound_embedding.parameters():
            param.requires_grad = True
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_head_parameters(self):
        params = list(self.error_detector.parameters()) + \
                 list(self.action_head.parameters())
        for head in self.sub_heads:
            params.extend(head.parameters())
        return params

    def get_bert_parameters(self):
        return list(self.compound_embedding.parameters()) + \
               list(self.bert.parameters())


def load_pretrained_bert(checkpoint_path, model):
    """
    Load pretrained CPBertForMaskedLM weights into CPMusicGECToR.

    Loads compound_embedding and bert encoder weights, ignores MLM heads.
    """
    model_path = os.path.join(checkpoint_path, 'model.pt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        from safetensors.torch import load_file
        safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    # Load compound_embedding weights
    ce_state = {}
    bert_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k.startswith('compound_embedding.'):
            ce_state[k[len('compound_embedding.'):]] = v
        elif k.startswith('bert.'):
            bert_state[k[len('bert.'):]] = v
        else:
            skipped.append(k)

    missing_ce, unexpected_ce = model.compound_embedding.load_state_dict(ce_state, strict=False)
    missing_bert, unexpected_bert = model.bert.load_state_dict(bert_state, strict=False)

    if missing_ce or missing_bert:
        print(f"[Load] Missing: CE={missing_ce}, BERT={missing_bert}")
    if unexpected_ce or unexpected_bert:
        print(f"[Load] Unexpected: CE={unexpected_ce}, BERT={unexpected_bert}")
    if skipped:
        print(f"[Load] Skipped {len(skipped)} keys (MLM heads etc.)")

    return model


def compute_loss(detect_logits, action_logits, sub_logits,
                 detect_labels, action_labels, sub_token_labels,
                 attention_mask=None,
                 keep_weight=0.15, lambda_detect=0.5, lambda_sub=1.0):
    """
    Compute factored GECToR loss.

    Args:
        detect_logits: (batch, seq_len, 2)
        action_logits: (batch, seq_len, 4)
        sub_logits: list of 5 tensors, each (batch, seq_len, sub_vocab_size)
        detect_labels: (batch, seq_len) values 0/1/-100
        action_labels: (batch, seq_len) values 0-3/-100
        sub_token_labels: (batch, seq_len, 5) values or -100
        keep_weight: class weight for KEEP action (action 0)
        lambda_detect: weight for detection loss
        lambda_sub: weight for sub-token losses

    Returns:
        (total_loss, action_loss, sub_loss, detect_loss)
    """
    # Action loss with KEEP downweighting
    action_weights = torch.ones(NUM_ACTIONS, device=action_logits.device)
    action_weights[ACTION_KEEP] = keep_weight
    action_loss_fn = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-100)
    action_loss = action_loss_fn(
        action_logits.view(-1, NUM_ACTIONS),
        action_labels.view(-1),
    )

    # Sub-token losses (only for REPLACE/APPEND positions)
    sub_loss = torch.tensor(0.0, device=action_logits.device)
    sub_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for i, logits in enumerate(sub_logits):
        sub_labels_i = sub_token_labels[:, :, i]  # (batch, seq_len)
        sub_loss = sub_loss + sub_loss_fn(
            logits.view(-1, logits.size(-1)),
            sub_labels_i.view(-1),
        )

    # Detection loss
    detect_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    detect_loss = detect_loss_fn(
        detect_logits.view(-1, 2),
        detect_labels.view(-1),
    )

    total_loss = action_loss + lambda_sub * sub_loss + lambda_detect * detect_loss
    return total_loss, action_loss, sub_loss, detect_loss
