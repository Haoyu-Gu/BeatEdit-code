"""
FELIX-Music Inserter Training Script.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 conda run -n musictoken accelerate launch \
        --main_process_port=29501 --num_processes=2 --mixed_precision=fp16 \
        training/train_inserter.py --epochs 30 --batch_size 32 --gradient_accumulation 3

Features:
- Accelerate-based DDP training
- CrossEntropy loss on MASK positions only
- Cosine schedule with warmup
- Optional pretrained BERT weight loading
- Evaluation: top-1/top-5 accuracy at MASK positions
"""

import os
import sys

# Ensure FELIX root is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from configs.config import (
    InserterConfig, FELIXTrainingConfig, VOCAB_SIZE,
    PAD_TOKEN, MASK_TOKEN, DATA_DIR,
)
from data.dataset import (
    FELIXInserterDataset, FELIXInserterCollator, BucketBatchSampler, get_file_lists,
)
from models.inserter import FELIXInserter


def parse_args():
    parser = argparse.ArgumentParser(description='Train FELIX Inserter')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default='checkpoints/inserter')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained_bert', type=str, default=None,
                        help='Path to pretrained music_bert_with_pair checkpoint')
    parser.add_argument('--level_weights', type=str, default='30,30,25,15')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--tb_dir', type=str, default='logs/tb_inserter')
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_lengths_cache(data_dir, file_list):
    cache_path = os.path.join(data_dir, '.lengths_cache_with_pair.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_lengths = pickle.load(f)
        return [all_lengths.get(fname, 1000) for fname in file_list]
    return [1000] * len(file_list)


def compute_inserter_loss(logits, mask_targets, mask_positions):
    """
    Compute cross-entropy loss only at valid MASK positions.

    Args:
        logits: (B, M, V) predictions
        mask_targets: (B, M) target token IDs
        mask_positions: (B, M) position indices (-1 = padding)

    Returns:
        loss: scalar
    """
    valid = mask_positions >= 0  # (B, M)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits = logits[valid]       # (num_valid, V)
    valid_targets = mask_targets[valid]  # (num_valid,)

    loss = F.cross_entropy(valid_logits, valid_targets, ignore_index=PAD_TOKEN)
    return loss


def evaluate(model, eval_loader, accelerator):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_top5_correct = 0
    total_valid = 0

    with torch.no_grad():
        for batch in eval_loader:
            logits = model(
                batch['skeleton_ids'],
                batch['attention_mask'],
                batch['mask_positions'],
            )
            loss = compute_inserter_loss(logits, batch['mask_targets'], batch['mask_positions'])
            total_loss += loss.item()
            total_steps += 1

            # Accuracy
            valid = batch['mask_positions'] >= 0
            if valid.any():
                preds = logits[valid].argmax(dim=-1)
                targets = batch['mask_targets'][valid]
                total_correct += (preds == targets).sum().item()

                # Top-5
                top5 = logits[valid].topk(5, dim=-1).indices
                total_top5_correct += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()

                total_valid += valid.sum().item()

    avg_loss = total_loss / max(total_steps, 1)
    top1_acc = total_correct / max(total_valid, 1)
    top5_acc = total_top5_correct / max(total_valid, 1)

    model.train()
    return avg_loss, top1_acc, top5_acc


def main():
    args = parse_args()
    level_weights = tuple(int(x) for x in args.level_weights.split(','))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Data
    if accelerator.is_main_process:
        print("Loading file lists...")
    train_files, val_files, _ = get_file_lists(args.data_dir)
    if accelerator.is_main_process:
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = FELIXInserterDataset(
        train_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        pitch_shift_augment=True, level_weights=level_weights,
    )
    val_ds = FELIXInserterDataset(
        val_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        pitch_shift_augment=False, level_weights=level_weights,
    )

    train_lengths = load_lengths_cache(args.data_dir, train_files)
    train_sampler = BucketBatchSampler(
        train_lengths, batch_size=args.batch_size, bucket_size=200, shuffle=True
    )

    collator = FELIXInserterCollator(max_length=args.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collator, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=2,
        shuffle=False,
    )

    # Model
    config = InserterConfig()
    model = FELIXInserter(config)

    # Load pretrained BERT if specified
    if args.pretrained_bert:
        model.load_pretrained_bert(args.pretrained_bert)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Inserter model: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Schedule
    steps_per_epoch = len(train_sampler) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Load checkpoint
    start_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")

    # Prepare
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Advance scheduler to correct position for resumed training
    if start_epoch > 0:
        skip_steps = start_epoch * steps_per_epoch
        if accelerator.is_main_process:
            print(f"Advancing scheduler by {skip_steps} steps...")
        for _ in range(skip_steps):
            scheduler.step()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(args.tb_dir)

    # Training loop
    global_step = start_epoch * steps_per_epoch
    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                logits = model(
                    batch['skeleton_ids'],
                    batch['attention_mask'],
                    batch['mask_positions'],
                )
                loss = compute_inserter_loss(logits, batch['mask_targets'], batch['mask_positions'])
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if accelerator.is_main_process and global_step % args.log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step} | loss={avg_loss:.4f} | lr={lr:.2e}")
                if writer:
                    writer.add_scalar('train/loss_step', avg_loss, global_step)
                    writer.add_scalar('train/lr', lr, global_step)

        elapsed = time.time() - t_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} | train_loss={avg_train_loss:.4f} | time={elapsed:.0f}s")
            if writer:
                writer.add_scalar('train/loss', avg_train_loss, epoch + 1)

        # Evaluation
        val_loss, top1_acc, top5_acc = evaluate(model, val_loader, accelerator)

        if accelerator.is_main_process:
            print(f"  val_loss={val_loss:.4f} | top1_acc={top1_acc:.4f} | top5_acc={top5_acc:.4f}")
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch + 1)
                writer.add_scalar('val/top1_acc', top1_acc, epoch + 1)
                writer.add_scalar('val/top5_acc', top5_acc, epoch + 1)

        # Save
        if accelerator.is_main_process and (epoch + 1) % args.save_interval == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(args.output_dir, f'inserter_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'top1_acc': top1_acc,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            if top1_acc > best_val_acc:
                best_val_acc = top1_acc
                best_path = os.path.join(args.output_dir, 'inserter_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'val_loss': val_loss,
                    'top1_acc': top1_acc,
                }, best_path)
                print(f"  New best model! Top1={top1_acc:.4f}")

    if writer:
        writer.close()

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val top1: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
