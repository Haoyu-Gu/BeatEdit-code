"""
Training script for Levenshtein Transformer Music Inpainting.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n musictoken \
        python -u -m accelerate.commands.launch \
        --num_processes=2 --mixed_precision=fp16 \
        training/train.py --epochs 30 --batch_size 32 --gradient_accumulation 2

Features:
- Accelerate-based DDP training
- Three-head joint loss (deletion + insertion + token)
- Context-masked loss (only compute loss in mask region)
- Cosine schedule with warmup
- Checkpoint saving with best model tracking
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from configs.config import (
    LevTModelConfig, LevTTrainingConfig,
    PAD_TOKEN, PLH_TOKEN, VOCAB_SIZE, DATA_DIR,
)
from data.dataset import (
    LevTDataset, LevTCollator, BucketBatchSampler,
    get_file_lists, load_lengths_cache,
)
from models.levenshtein_transformer import LevenshteinTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Train LevT Music Inpainting')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--max_insert', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default='checkpoints/levt')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained_bert', type=str, default=None,
                        help='Path to pretrained BERT checkpoint for weight init')
    parser.add_argument('--w_del', type=float, default=1.0, help='Deletion loss weight')
    parser.add_argument('--w_ins', type=float, default=1.0, help='Insertion loss weight')
    parser.add_argument('--w_tok', type=float, default=1.0, help='Token loss weight')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mask_ratio_min', type=float, default=0.125)
    parser.add_argument('--mask_ratio_max', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=1)
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model, batch, w_del, w_ins, w_tok, label_smoothing, max_insert):
    """
    Compute combined loss from three heads.

    For deletion and insertion: run encoder on z_ids (intermediate state).
    For token prediction: run encoder on tok_input_ids (intermediate with PLH
    tokens inserted at gaps indicated by ins_labels), then compute CE loss
    only at PLH positions against the correct target tokens.
    """
    z_ids = batch['z_ids']                   # (B, L)
    attention_mask = batch['attention_mask']  # (B, L)

    # Forward pass on z_ids for deletion and insertion heads
    model_output = model(z_ids, attention_mask, operation='all')

    del_logits = model_output['del_logits']  # (B, L, 2)
    ins_logits = model_output['ins_logits']  # (B, L+1, max_insert+1)

    del_labels = batch['del_labels']         # (B, L)
    ins_labels = batch['ins_labels']         # (B, L+1)
    context_mask = batch['context_mask']     # (B, L) 1=editable, 0=frozen

    B, L = del_logits.shape[:2]
    device = del_logits.device

    # ---- Deletion Loss ----
    valid_del_mask = (del_labels != -100)  # (B, L)
    if valid_del_mask.any():
        del_logits_flat = del_logits.view(-1, 2)
        del_labels_flat = del_labels.view(-1)
        del_loss = F.cross_entropy(
            del_logits_flat, del_labels_flat,
            ignore_index=-100, reduction='mean',
        )
    else:
        del_loss = torch.tensor(0.0, device=device)

    # ---- Insertion Loss ----
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

        ins_loss_all = F.cross_entropy(
            ins_logits_flat, ins_labels_flat, reduction='none'
        )
        ins_loss = (ins_loss_all * ins_valid_flat.float()).sum() / ins_valid_flat.float().sum().clamp(min=1)
    else:
        ins_loss = torch.tensor(0.0, device=device)

    # ---- Token Loss ----
    # Run token head on tok_input_ids (intermediate with PLH at insertion points).
    # Gather logits at PLH positions and compute CE against tok_targets.
    tok_input_ids = batch['tok_input_ids']        # (B, L2)
    tok_attention_mask = batch['tok_attention_mask']  # (B, L2)
    tok_positions = batch['tok_positions']         # (B, M)
    tok_targets = batch['tok_targets']             # (B, M)

    if tok_positions.numel() > 0 and (tok_positions >= 0).any():
        # Separate forward pass through encoder + token head on tok_input_ids
        tok_logits_full = model.forward_tok(tok_input_ids, tok_attention_mask)  # (B, L2, V)

        B2, M = tok_positions.shape
        V = tok_logits_full.size(-1)
        safe_pos = tok_positions.clamp(min=0)  # (B, M)
        indices = safe_pos.unsqueeze(-1).expand(B2, M, V)
        tok_logits_at_plh = torch.gather(tok_logits_full, dim=1, index=indices)  # (B, M, V)

        # Mask out padding positions (padded with -1)
        valid_mask = (tok_positions >= 0)  # (B, M)

        tok_logits_flat = tok_logits_at_plh.reshape(-1, V)
        tok_targets_flat = tok_targets.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        tok_loss_all = F.cross_entropy(
            tok_logits_flat, tok_targets_flat,
            ignore_index=PAD_TOKEN,
            label_smoothing=label_smoothing,
            reduction='none',
        )
        tok_loss = (tok_loss_all * valid_flat.float()).sum() / valid_flat.float().sum().clamp(min=1)
    else:
        tok_loss = torch.tensor(0.0, device=device)

    # Combined loss
    total_loss = w_del * del_loss + w_ins * ins_loss + w_tok * tok_loss

    return total_loss, del_loss.item(), ins_loss.item(), tok_loss.item()


def evaluate(model, eval_loader, args, accelerator):
    """Run evaluation."""
    model.eval()
    total_loss = 0.0
    total_del_loss = 0.0
    total_ins_loss = 0.0
    total_tok_loss = 0.0
    total_steps = 0

    # Deletion accuracy
    del_correct = 0
    del_total = 0

    # Insertion accuracy
    ins_correct = 0
    ins_total = 0

    with torch.no_grad():
        for batch in eval_loader:
            loss, dl, il, tl = compute_loss(
                model, batch,
                args.w_del, args.w_ins, args.w_tok,
                args.label_smoothing, args.max_insert,
            )

            total_loss += loss.item()
            total_del_loss += dl
            total_ins_loss += il
            total_tok_loss += tl
            total_steps += 1

            # Deletion accuracy (separate forward for metrics)
            output = model(batch['z_ids'], batch['attention_mask'], operation='all')
            del_preds = output['del_logits'].argmax(dim=-1)
            del_labels = batch['del_labels']
            valid = del_labels != -100
            if valid.any():
                del_correct += (del_preds[valid] == del_labels[valid]).sum().item()
                del_total += valid.sum().item()

            # Insertion accuracy (exact count match)
            ins_preds = output['ins_logits'].argmax(dim=-1)
            ins_labels = batch['ins_labels']
            ins_L = min(ins_preds.size(1), ins_labels.size(1))
            ins_valid = batch['attention_mask'][:, :ins_L - 1] if ins_L > 1 else torch.ones(1)
            if ins_L > 1:
                ins_correct += (ins_preds[:, :ins_L] == ins_labels[:, :ins_L]).float().sum().item()
                ins_total += ins_L * ins_labels.size(0)

    model.train()

    n = max(total_steps, 1)
    metrics = {
        'loss': total_loss / n,
        'del_loss': total_del_loss / n,
        'ins_loss': total_ins_loss / n,
        'tok_loss': total_tok_loss / n,
        'del_acc': del_correct / max(del_total, 1),
        'ins_acc': ins_correct / max(ins_total, 1),
    }
    return metrics


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Data
    if accelerator.is_main_process:
        print("Loading file lists...")
    train_files, val_files, _ = get_file_lists(args.data_dir)
    if accelerator.is_main_process:
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = LevTDataset(
        train_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
        pitch_shift_augment=True, max_insert=args.max_insert,
    )
    val_ds = LevTDataset(
        val_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
        pitch_shift_augment=False, max_insert=args.max_insert,
    )

    # Lengths for bucket sampling
    train_lengths = load_lengths_cache(args.data_dir, train_files)
    train_sampler = BucketBatchSampler(
        train_lengths, batch_size=args.batch_size, bucket_size=200, shuffle=True
    )

    collator = LevTCollator(max_length=args.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collator, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=2,
        shuffle=False,
    )

    # Model
    model_config = LevTModelConfig(max_insert=args.max_insert)
    model = LevenshteinTransformer(model_config)

    # Optionally load pretrained BERT weights
    if args.pretrained_bert:
        model.load_pretrained_bert(args.pretrained_bert)

    if accelerator.is_main_process:
        total, trainable = model.count_parameters()
        print(f"LevT model: {total:,} total params ({total/1e6:.1f}M), "
              f"{trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Schedule
    steps_per_epoch = len(train_sampler) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Load checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Output dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_del = 0.0
        epoch_ins = 0.0
        epoch_tok = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, dl, il, tl = compute_loss(
                    model, batch,
                    args.w_del, args.w_ins, args.w_tok,
                    args.label_smoothing, args.max_insert,
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

        # Epoch summary
        elapsed = time.time() - t_start
        n = max(epoch_steps, 1)

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} | "
                  f"loss={epoch_loss/n:.4f} del={epoch_del/n:.4f} "
                  f"ins={epoch_ins/n:.4f} tok={epoch_tok/n:.4f} | "
                  f"time={elapsed:.0f}s")

        # Evaluation
        metrics = evaluate(model, val_loader, args, accelerator)

        if accelerator.is_main_process:
            print(f"  val: loss={metrics['loss']:.4f} "
                  f"del_loss={metrics['del_loss']:.4f} "
                  f"ins_loss={metrics['ins_loss']:.4f} "
                  f"tok_loss={metrics['tok_loss']:.4f}")
            print(f"  del_acc={metrics['del_acc']:.4f} "
                  f"ins_acc={metrics['ins_acc']:.4f}")

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_interval == 0:
            unwrapped = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(args.output_dir, f'levt_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': metrics['loss'],
                'config': model_config.__dict__,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                best_path = os.path.join(args.output_dir, 'levt_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped.state_dict(),
                    'val_loss': metrics['loss'],
                    'config': model_config.__dict__,
                }, best_path)
                print(f"  New best model! loss={metrics['loss']:.4f}")

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
