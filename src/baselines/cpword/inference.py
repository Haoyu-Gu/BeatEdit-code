"""
CPWord Music GECToR Inference Pipeline

Multi-round iterative inference with factored heads:
- Action prediction (KEEP/DELETE/REPLACE/APPEND)
- Sub-token prediction for REPLACE/APPEND
- KEEP confidence bias + error detection thresholding
- Post-processing for CPWord structure validity
"""

import os
import argparse
import torch
import numpy as np
from typing import List

from config import (
    NUM_ACTIONS, ACTION_KEEP, ACTION_DELETE, ACTION_REPLACE, ACTION_APPEND,
    SUB_VOCAB_SIZES,
    FAMILY_METRIC, FAMILY_NOTE,
    POS_BAR, POS_OFFSET, POS_MAX, POS_IGNORE,
    PITCH_IGNORE, PITCH_MIN_ID, PITCH_MAX_ID, PITCH_OFFSET, MIDI_PITCH_MIN,
    VEL_IGNORE, VEL_OFFSET, VEL_MAX_ID,
    DUR_IGNORE, DUR_OFFSET, DUR_MAX_ID,
    BOS_TOKEN, EOS_TOKEN,
    is_special_token, is_bar_token, is_position_token, is_note_token,
    is_music_token,
    TRAINING_DEFAULTS as TD,
)
from sequence_parser import parse_sequence, bars_to_flat_notes, flat_notes_to_bars, reassemble_sequence
from model import CPMusicGECToR


def apply_actions(tokens, actions, new_sub_tokens):
    """
    Apply predicted actions and sub-tokens to compound token sequence.

    Args:
        tokens: list of compound tokens [family, pos, pitch, vel, dur]
        actions: list of action IDs (0-3)
        new_sub_tokens: list of [5] sub-token predictions

    Returns:
        new token sequence
    """
    result = []

    for tok, action, new_subs in zip(tokens, actions, new_sub_tokens):
        if action == ACTION_KEEP:
            result.append(list(tok))
        elif action == ACTION_DELETE:
            pass  # skip
        elif action == ACTION_REPLACE:
            result.append(list(new_subs))
        elif action == ACTION_APPEND:
            result.append(list(tok))      # keep current
            result.append(list(new_subs)) # insert new after

    return result


def validate_compound_token(tok):
    """Validate and clamp sub-token IDs to valid ranges."""
    tok = list(tok)
    if tok[0] == FAMILY_NOTE:
        # Note: [Note, Ignore, Pitch, Vel, Dur]
        tok[1] = POS_IGNORE
        tok[2] = max(PITCH_MIN_ID, min(tok[2], PITCH_MAX_ID))
        tok[3] = max(VEL_OFFSET, min(tok[3], VEL_MAX_ID))
        tok[4] = max(DUR_OFFSET, min(tok[4], DUR_MAX_ID))
    elif tok[0] == FAMILY_METRIC:
        if tok[1] == POS_BAR:
            # Bar token
            tok[2] = PITCH_IGNORE
            tok[3] = VEL_IGNORE
            tok[4] = DUR_IGNORE
        elif POS_OFFSET <= tok[1] <= POS_MAX:
            # Position token
            tok[2] = PITCH_IGNORE
            tok[3] = VEL_IGNORE
            tok[4] = DUR_IGNORE
    return tok


def post_process(tokens):
    """
    Post-process edited CPWord tokens to ensure structural validity.

    Validates:
    1. Bar -> Position -> Note structure
    2. Sub-token values in valid ranges
    3. No duplicate pitches at same position
    """
    try:
        parsed = parse_sequence(tokens)
    except Exception:
        return tokens

    new_bars = []
    for bar in parsed['bars']:
        bar_data = []
        for pos in bar['positions']:
            valid_notes = []
            seen_pitches = set()
            for note in pos['notes']:
                p = note['pitch_id']
                v = note['velocity_id']
                d = note['duration_id']
                if PITCH_MIN_ID <= p <= PITCH_MAX_ID and \
                   VEL_OFFSET <= v <= VEL_MAX_ID and \
                   DUR_OFFSET <= d <= DUR_MAX_ID and \
                   p not in seen_pitches:
                    seen_pitches.add(p)
                    valid_notes.append({
                        'pitch_id': p,
                        'velocity_id': v,
                        'duration_id': d,
                    })
            if valid_notes:
                bar_data.append({
                    'position_value': pos['position_value'],
                    'notes': valid_notes,
                })
        new_bars.append(bar_data)

    return reassemble_sequence(parsed, new_bars)


def inference_single(model, input_tokens, device='cpu',
                     max_iterations=3, keep_confidence_bias=0.3,
                     error_threshold=0.5):
    """
    GECToR-style iterative inference for a single CPWord sequence.
    """
    model.eval()
    current_tokens = [list(t) for t in input_tokens]
    info = {'iterations': 0, 'edits_per_round': []}

    for iteration in range(max_iterations):
        if len(current_tokens) > 2048:
            current_tokens = current_tokens[:2048]

        compound_ids = torch.tensor([current_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones(1, len(current_tokens), dtype=torch.long, device=device)

        with torch.no_grad():
            detect_logits, action_logits, sub_logits = model(compound_ids, attention_mask)

        detect_probs = torch.softmax(detect_logits[0], dim=-1)
        action_logits_seq = action_logits[0]

        # Apply KEEP confidence bias
        action_logits_seq[:, ACTION_KEEP] += keep_confidence_bias

        predicted_actions = []
        predicted_subs = []
        num_edits = 0

        for i in range(len(current_tokens)):
            if is_special_token(current_tokens[i]):
                predicted_actions.append(ACTION_KEEP)
                predicted_subs.append(list(current_tokens[i]))
                continue

            error_prob = detect_probs[i, 1].item()
            if error_prob < error_threshold:
                predicted_actions.append(ACTION_KEEP)
                predicted_subs.append(list(current_tokens[i]))
            else:
                action = action_logits_seq[i].argmax().item()
                predicted_actions.append(action)

                if action in (ACTION_REPLACE, ACTION_APPEND):
                    # Predict sub-tokens
                    new_tok = []
                    for j in range(5):
                        sub_pred = sub_logits[j][0, i].argmax().item()
                        new_tok.append(sub_pred)
                    new_tok = validate_compound_token(new_tok)
                    predicted_subs.append(new_tok)
                else:
                    predicted_subs.append(list(current_tokens[i]))

                if action != ACTION_KEEP:
                    num_edits += 1

        info['edits_per_round'].append(num_edits)
        info['iterations'] = iteration + 1

        if num_edits == 0:
            break

        new_tokens = apply_actions(current_tokens, predicted_actions, predicted_subs)
        new_tokens = post_process(new_tokens)

        if new_tokens == current_tokens:
            break

        current_tokens = new_tokens

    return current_tokens, info


def inference_batch(model, token_sequences, device='cpu', batch_size=32,
                    max_iterations=3, keep_confidence_bias=0.3,
                    error_threshold=0.5):
    """Batch inference for multiple sequences."""
    results = []
    infos = []

    for i in range(0, len(token_sequences), batch_size):
        batch_seqs = token_sequences[i:i + batch_size]
        for seq in batch_seqs:
            corrected, info = inference_single(
                model, seq, device=device,
                max_iterations=max_iterations,
                keep_confidence_bias=keep_confidence_bias,
                error_threshold=error_threshold,
            )
            results.append(corrected)
            infos.append(info)

    return results, infos


def load_model_for_inference(checkpoint_path, device='cpu'):
    """Load trained CPMusicGECToR model for inference."""
    model = CPMusicGECToR(
        sub_vocab_sizes=SUB_VOCAB_SIZES,
        hidden_size=TD['hidden_size'],
        dropout=TD['dropout'],
    )

    model_path = os.path.join(checkpoint_path, 'model.pt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.to(device)
    model.eval()
    return model
