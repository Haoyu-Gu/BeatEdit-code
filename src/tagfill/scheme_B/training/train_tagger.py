"""
FELIX-Music Tagger Training Script.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 conda run -n musictoken accelerate launch \
        --num_processes=2 --mixed_precision=fp16 \
        training/train_tagger.py --epochs 30 --batch_size 32 --gradient_accumulation 3

Features:
- Accelerate-based DDP training
- Focal Loss for class imbalance
- Cosine schedule with warmup
- Per-label evaluation metrics (accuracy, macro-F1)
- Length-aware bucket sampling
- Checkpoint saving
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
from sklearn.metrics import f1_score, classification_report

from configs.config import (
    TaggerConfig, FELIXTrainingConfig, NUM_FELIX_LABELS,
    PAD_TOKEN, LABEL_PAD, DATA_DIR, decode_felix_label,
)
from data.dataset import (
    FELIXTaggerDataset, FELIXTaggerCollator, BucketBatchSampler, get_file_lists,
)
from models.tagger import FELIXTagger, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train FELIX Tagger')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default='checkpoints/tagger')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained_bert', type=str, default=None,
                        help='Path to pretrained Music BERT checkpoint for weight init')
    parser.add_argument('--level_weights', type=str, default='30,30,25,15',
                        help='Perturbation level weights L1,L2,L3,L4')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=1, help='Save every N epochs')
    parser.add_argument('--tb_dir', type=str, default='logs/tb_tagger')
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_lengths_cache(data_dir, file_list):
    """Try to load cached lengths or estimate."""
    cache_path = os.path.join(data_dir, '.lengths_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_lengths = pickle.load(f)
        lengths = []
        for fname in file_list:
            lengths.append(all_lengths.get(fname, 1000))
        return lengths
    return [1000] * len(file_list)


def evaluate(model, eval_loader, loss_fn, accelerator):
    """Run evaluation and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            logits = model(
                batch['input_ids'],
                batch['attention_mask'],
            )
            loss = loss_fn(logits, batch['labels'])
            total_loss += loss.item()
            total_steps += 1

            # Collect predictions
            preds = logits.argmax(dim=-1)  # (B, L)
            labels = batch['labels']       # (B, L)

            # Flatten and filter padding
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            valid = labels_flat != LABEL_PAD
            all_preds.extend(preds_flat[valid].tolist())
            all_labels.extend(labels_flat[valid].tolist())

    avg_loss = total_loss / max(total_steps, 1)

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-category accuracy
    category_acc = {}
    for lid in range(NUM_FELIX_LABELS):
        mask = all_labels == lid
        if mask.sum() > 0:
            op, val = decode_felix_label(lid)
            cat_acc = (all_preds[mask] == lid).mean()
            category_acc[f'{op}({val})'] = (cat_acc, int(mask.sum()))

    model.train()
    return avg_loss, accuracy, macro_f1, category_acc


def main():
    args = parse_args()
    level_weights = tuple(int(x) for x in args.level_weights.split(','))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Data
    if accelerator.is_main_process:
        print("Loading file lists...")
    train_files, val_files, test_files = get_file_lists(args.data_dir)
    if accelerator.is_main_process:
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_ds = FELIXTaggerDataset(
        train_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        pitch_shift_augment=True, level_weights=level_weights,
    )
    val_ds = FELIXTaggerDataset(
        val_files, data_dir=args.data_dir, max_len=args.max_seq_len,
        pitch_shift_augment=False, level_weights=level_weights,
    )

    # Lengths for bucket sampling
    train_lengths = load_lengths_cache(args.data_dir, train_files)
    train_sampler = BucketBatchSampler(
        train_lengths, batch_size=args.batch_size, bucket_size=200, shuffle=True
    )

    collator = FELIXTaggerCollator(max_length=args.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collator, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=2,
        shuffle=False,
    )

    # Model
    config = TaggerConfig()
    model = FELIXTagger(config)

    # Load pretrained BERT weights (§3.1: shared pre-trained backbone)
    if args.pretrained_bert:
        model.load_pretrained_bert(args.pretrained_bert)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Tagger model: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Schedule
    steps_per_epoch = len(train_sampler) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss
    loss_fn = FocalLoss(gamma=args.focal_gamma, ignore_index=LABEL_PAD)

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")

    # Prepare with accelerator
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

    # Output dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(args.tb_dir)

    # Training loop
    global_step = start_epoch * steps_per_epoch
    best_val_f1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                logits = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                )
                loss = loss_fn(logits, batch['labels'])
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

        # Epoch summary
        elapsed = time.time() - t_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} | train_loss={avg_train_loss:.4f} | time={elapsed:.0f}s")
            if writer:
                writer.add_scalar('train/loss', avg_train_loss, epoch + 1)

        # Evaluation
        val_loss, val_acc, val_f1, cat_acc = evaluate(
            model, val_loader, loss_fn, accelerator
        )

        if accelerator.is_main_process:
            print(f"  val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}")
            for cat, (acc, cnt) in sorted(cat_acc.items(), key=lambda x: -x[1][1])[:10]:
                print(f"    {cat}: acc={acc:.3f} (n={cnt})")
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch + 1)
                writer.add_scalar('val/acc', val_acc, epoch + 1)
                writer.add_scalar('val/macro_f1', val_f1, epoch + 1)

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_interval == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(args.output_dir, f'tagger_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_path = os.path.join(args.output_dir, 'tagger_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                }, best_path)
                print(f"  New best model! F1={val_f1:.4f}")

    if writer:
        writer.close()

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")


if __name__ == '__main__':
    main()
