"""
Training script for Levenshtein Transformer Accompaniment-Only Inpainting.

Trains LevT to generate accompaniment conditioned on melody context.
Unlike vanilla inpainting (masks entire beats), this only masks accomp tokens.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python -u -m accelerate.commands.launch \
        --num_processes=4 --mixed_precision=fp16 \
        LevT_inpainting/training/train_accomp_inpainting.py \
        --scheme A \
        --checkpoint LevT_training_results/vanilla/scheme_a/levt_best.pt \
        --lr 1e-4 --batch_size 16 --epochs 5 --gradient_accumulation 4 \
        --output_dir LevT_training_results/accomp_inpainting/scheme_a
"""

import os
import sys
import time
import argparse
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Project root
LEVT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, LEVT_DIR)

from models.levenshtein_transformer import LevenshteinTransformer
from configs.config import LevTModelConfig
from data.dataset_accomp_inpainting import (
    LevTAccompInpaintingDataset, get_file_lists, load_lengths_cache,
    SCHEME_TOKENS, DATA_DIR,
)
from data.dataset_editing import LevTEditingCollator, BucketBatchSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train LevT Editing')
    parser.add_argument('--scheme', type=str, default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--max_insert', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default='LevT_training_results/editing_v2/scheme_a')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Vanilla checkpoint to initialize from')
    parser.add_argument('--w_del', type=float, default=1.0)
    parser.add_argument('--w_ins', type=float, default=1.0)
    parser.add_argument('--w_tok', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=1)
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model, batch, w_del, w_ins, w_tok, label_smoothing, max_insert, pad_token_id, raw_model=None):
    """Compute combined three-head loss.
    raw_model: unwrapped model for calling sub-methods (forward_tok) under DDP.
    """
    z_ids = batch['z_ids']
    attention_mask = batch['attention_mask']
    if raw_model is None:
        raw_model = model

    model_output = model(z_ids, attention_mask, operation='all')
    del_logits = model_output['del_logits']
    ins_logits = model_output['ins_logits']

    del_labels = batch['del_labels']
    ins_labels = batch['ins_labels']

    B, L = del_logits.shape[:2]
    device = del_logits.device

    # Deletion Loss
    valid_del_mask = (del_labels != -100)
    if valid_del_mask.any():
        del_loss = F.cross_entropy(
            del_logits.view(-1, 2), del_labels.view(-1),
            ignore_index=-100, reduction='mean',
        )
    else:
        del_loss = torch.tensor(0.0, device=device)

    # Insertion Loss
    ins_L = min(ins_logits.size(1), ins_labels.size(1))
    ins_logits_trim = ins_logits[:, :ins_L]
    ins_labels_trim = ins_labels[:, :ins_L].clamp(0, max_insert)

    ins_valid = torch.ones(B, ins_L, device=device)
    if ins_L > 1:
        ins_valid[:, 1:] = attention_mask[:, :ins_L - 1].float()

    if ins_valid.any():
        ins_logits_flat = ins_logits_trim.reshape(-1, ins_logits_trim.size(-1))
        ins_labels_flat = ins_labels_trim.reshape(-1)
        ins_valid_flat = ins_valid.reshape(-1).bool()
        ins_loss_all = F.cross_entropy(ins_logits_flat, ins_labels_flat, reduction='none')
        ins_loss = (ins_loss_all * ins_valid_flat.float()).sum() / ins_valid_flat.float().sum().clamp(min=1)
    else:
        ins_loss = torch.tensor(0.0, device=device)

    # Token Loss
    tok_input_ids = batch['tok_input_ids']
    tok_attention_mask = batch['tok_attention_mask']
    tok_positions = batch['tok_positions']
    tok_targets = batch['tok_targets']

    if tok_positions.numel() > 0 and (tok_positions >= 0).any():
        tok_logits_full = raw_model.forward_tok(tok_input_ids, tok_attention_mask)
        B2, M = tok_positions.shape
        V = tok_logits_full.size(-1)
        safe_pos = tok_positions.clamp(min=0)
        indices = safe_pos.unsqueeze(-1).expand(B2, M, V)
        tok_logits_at_plh = torch.gather(tok_logits_full, dim=1, index=indices)
        valid_mask = (tok_positions >= 0)
        tok_logits_flat = tok_logits_at_plh.reshape(-1, V)
        tok_targets_flat = tok_targets.reshape(-1)
        valid_flat = valid_mask.reshape(-1)
        tok_loss_all = F.cross_entropy(
            tok_logits_flat, tok_targets_flat,
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
            reduction='none',
        )
        tok_loss = (tok_loss_all * valid_flat.float()).sum() / valid_flat.float().sum().clamp(min=1)
    else:
        tok_loss = torch.tensor(0.0, device=device)

    total_loss = w_del * del_loss + w_ins * ins_loss + w_tok * tok_loss
    return total_loss, del_loss.item(), ins_loss.item(), tok_loss.item()


def evaluate(model, eval_loader, args, accelerator, pad_token_id):
    model.eval()
    raw_model = accelerator.unwrap_model(model)
    total_loss = total_del = total_ins = total_tok = 0.0
    total_steps = 0
    del_correct = del_total = ins_correct = ins_total = 0

    with torch.no_grad():
        for batch in eval_loader:
            loss, dl, il, tl = compute_loss(
                model, batch,
                args.w_del, args.w_ins, args.w_tok,
                args.label_smoothing, args.max_insert, pad_token_id,
                raw_model=raw_model,
            )
            total_loss += loss.item()
            total_del += dl
            total_ins += il
            total_tok += tl
            total_steps += 1

            output = model(batch['z_ids'], batch['attention_mask'], operation='all')

            # Deletion accuracy (editable positions only)
            del_preds = output['del_logits'].argmax(dim=-1)
            del_labels = batch['del_labels']
            valid = del_labels != -100
            if valid.any():
                del_correct += (del_preds[valid] == del_labels[valid]).sum().item()
                del_total += valid.sum().item()

            # Insertion accuracy
            ins_preds = output['ins_logits'].argmax(dim=-1)
            ins_labels = batch['ins_labels']
            ins_L = min(ins_preds.size(1), ins_labels.size(1))
            if ins_L > 1:
                ins_correct += (ins_preds[:, :ins_L] == ins_labels[:, :ins_L]).float().sum().item()
                ins_total += ins_L * ins_labels.size(0)

    model.train()
    n = max(total_steps, 1)
    return {
        'loss': total_loss / n,
        'del_loss': total_del / n,
        'ins_loss': total_ins / n,
        'tok_loss': total_tok / n,
        'del_acc': del_correct / max(del_total, 1),
        'ins_acc': ins_correct / max(ins_total, 1),
    }


def main():
    args = parse_args()
    scheme = args.scheme
    tok = SCHEME_TOKENS[scheme]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Data
    if accelerator.is_main_process:
        print(f"=== LevT Accomp Inpainting Training (Scheme {scheme}) ===")
        print(f"Loading file lists from {args.data_dir}...")

    train_files, val_files, _ = get_file_lists(args.data_dir)

    if accelerator.is_main_process:
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")
        print(f"Scheme {scheme}: vocab={tok['vocab_size']}, pad={tok['pad']}, plh={tok['plh']}")

    train_ds = LevTAccompInpaintingDataset(
        train_files, scheme=scheme, data_dir=args.data_dir,
        max_len=args.max_seq_len, pitch_shift_augment=True,
        max_insert=args.max_insert,
    )
    val_ds = LevTAccompInpaintingDataset(
        val_files, scheme=scheme, data_dir=args.data_dir,
        max_len=args.max_seq_len, pitch_shift_augment=False,
        max_insert=args.max_insert,
    )

    # Lengths for bucket sampling
    train_lengths = load_lengths_cache(args.data_dir, train_files, scheme=scheme)
    train_sampler = BucketBatchSampler(
        train_lengths, batch_size=args.batch_size, bucket_size=200, shuffle=True
    )

    collator = LevTEditingCollator(pad_token_id=tok['pad'], max_length=args.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collator, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=2,
        shuffle=False,
    )

    # Model — use scheme-specific config
    model_config = LevTModelConfig(
        vocab_size=tok['vocab_size'],
        max_insert=args.max_insert,
        pad_token_id=tok['pad'],
        bos_token_id=tok['bos'],
        eos_token_id=tok['eos'],
        plh_token_id=tok['plh'],
    )
    model = LevenshteinTransformer(model_config)

    # Load vanilla checkpoint (weights only, not optimizer state)
    if args.checkpoint:
        if accelerator.is_main_process:
            print(f"Loading vanilla checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        # Remove track_bias if present (vanilla shouldn't have it)
        state_dict.pop('track_bias', None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            vanilla_loss = ckpt.get('val_loss', 'N/A')
            vanilla_epoch = ckpt.get('epoch', 'N/A')
            print(f"  Vanilla checkpoint: epoch={vanilla_epoch}, val_loss={vanilla_loss}")

    if accelerator.is_main_process:
        total, trainable = model.count_parameters()
        print(f"Model: {total:,} params ({total/1e6:.1f}M), {trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Schedule
    steps_per_epoch = len(train_sampler) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if accelerator.is_main_process:
        print(f"Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
              f"{total_steps} total steps, {warmup_steps} warmup")
        print(f"lr={args.lr}, batch_size={args.batch_size}, "
              f"grad_accum={args.gradient_accumulation}")

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Output dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    raw_model = accelerator.unwrap_model(model)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = epoch_del = epoch_ins = epoch_tok = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, dl, il, tl = compute_loss(
                    model, batch,
                    args.w_del, args.w_ins, args.w_tok,
                    args.label_smoothing, args.max_insert, tok['pad'],
                    raw_model=raw_model,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_del += dl
            epoch_ins += il
            epoch_tok += tl
            epoch_steps += 1
            global_step += 1

            if accelerator.is_main_process and global_step % args.log_interval == 0:
                n = max(epoch_steps, 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step} | loss={epoch_loss/n:.4f} "
                      f"del={epoch_del/n:.4f} ins={epoch_ins/n:.4f} tok={epoch_tok/n:.4f} "
                      f"| lr={lr:.2e}")

        elapsed = time.time() - t_start
        n = max(epoch_steps, 1)
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} | "
                  f"loss={epoch_loss/n:.4f} del={epoch_del/n:.4f} "
                  f"ins={epoch_ins/n:.4f} tok={epoch_tok/n:.4f} | "
                  f"time={elapsed:.0f}s")

        # Evaluation
        metrics = evaluate(model, val_loader, args, accelerator, tok['pad'])
        if accelerator.is_main_process:
            print(f"  val: loss={metrics['loss']:.4f} "
                  f"del={metrics['del_loss']:.4f} ins={metrics['ins_loss']:.4f} "
                  f"tok={metrics['tok_loss']:.4f}")
            print(f"  del_acc={metrics['del_acc']:.4f} ins_acc={metrics['ins_acc']:.4f}")

        # Save
        if accelerator.is_main_process and (epoch + 1) % args.save_interval == 0:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(args.output_dir, f'levt_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': metrics['loss'],
                'config': dataclasses.asdict(model_config),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                best_path = os.path.join(args.output_dir, 'levt_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped.state_dict(),
                    'val_loss': metrics['loss'],
                    'config': dataclasses.asdict(model_config),
                }, best_path)
                print(f"  *** New best model! loss={metrics['loss']:.4f} ***")

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
