"""
Music GECToR Training Script (absolute_bundled encoding - Scheme D)

Accelerate-based multi-GPU training with:
- Stage I: Synthetic error data (freeze BERT → unfreeze → fine-tune)
- Stage III: Mixed clean sample fine-tuning (prevent over-correction)

Usage:
    # Stage I training
    accelerate launch train_gector.py --stage 1 --epochs 20

    # Stage III fine-tuning
    accelerate launch train_gector.py --stage 3 --epochs 3 \
        --checkpoint checkpoints/gector_absolute_bundled/best_model \
        --clean_ratio 0.25 --lr 5e-6

    # Debug (single GPU, small data)
    python train_gector.py --stage 1 --epochs 1 --batch_size 4 \
        --max_samples 100 --debug
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import (
    NUM_LABELS, PAD_TOKEN, LABEL_PAD, DATA_DIR, VOCAB_SIZE,
    TRAINING_DEFAULTS as TD,
)
from model import MusicGECToR, load_pretrained_bert, compute_loss
from dataset import (
    GECToRDataset, GECToRCollator, BucketBatchSampler, get_file_lists,
)


BERT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'music_bert_absolute_bundled', 'checkpoints', 'music_bert_absolute_bundled', 'best_model'
)


# fp16 needs CUDA; fall back to full precision elsewhere so a small pilot run
# works on CPU/MPS. Override with BEATEDIT_PRECISION=no|fp16|bf16.
def _resolve_precision(configured='fp16'):
    requested = os.environ.get('BEATEDIT_PRECISION', configured)
    if requested in ('fp16', 'bf16') and not torch.cuda.is_available():
        return 'no'
    return requested


def parse_args():
    parser = argparse.ArgumentParser(description='Music GECToR Training (absolute_bundled)')
    # Stage
    parser.add_argument('--stage', type=int, default=1, choices=[1, 3],
                        help='Training stage: 1=synthetic, 3=clean mixing')
    # Data
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--max_len', type=int, default=TD['max_seq_len'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (for debugging)')
    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from GECToR checkpoint')
    parser.add_argument('--bert_checkpoint', type=str, default=BERT_CHECKPOINT,
                        help='Pretrained BERT checkpoint path')
    # Training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=TD['batch_size_per_gpu'])
    parser.add_argument('--freeze_epochs', type=int, default=TD['freeze_epochs'])
    parser.add_argument('--cold_lr', type=float, default=TD['cold_lr'])
    parser.add_argument('--lr_bert', type=float, default=TD['finetune_lr_bert'])
    parser.add_argument('--lr_head', type=float, default=TD['finetune_lr_head'])
    parser.add_argument('--weight_decay', type=float, default=TD['weight_decay'])
    parser.add_argument('--warmup_ratio', type=float, default=TD['warmup_ratio'])
    parser.add_argument('--keep_weight', type=float, default=TD['keep_weight'])
    parser.add_argument('--lambda_detect', type=float, default=TD['lambda_detect'])
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    # Stage III
    parser.add_argument('--clean_ratio', type=float, default=TD['clean_ratio'])
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (for Stage III)')
    # Perturbation
    parser.add_argument('--p_pitch', type=float, default=0.10)
    parser.add_argument('--p_rhythm', type=float, default=0.05)
    parser.add_argument('--p_delete', type=float, default=0.03)
    parser.add_argument('--p_insert', type=float, default=0.02)
    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gector_absolute_bundled')
    parser.add_argument('--patience', type=int, default=TD['early_stopping_patience'])
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Set defaults based on stage
    if args.epochs is None:
        args.epochs = TD['stage1_epochs'] if args.stage == 1 else TD['stage3_epochs']

    return args


def build_model(args):
    """Create MusicGECToR model with pretrained BERT."""
    model = MusicGECToR(num_labels=NUM_LABELS, dropout=TD['dropout'])

    if args.checkpoint:
        # Resume from GECToR checkpoint
        state_dict = torch.load(
            os.path.join(args.checkpoint, 'model.pt'),
            map_location='cpu',
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Checkpoint] Missing keys: {missing}")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys (ignored): {unexpected}")
        print(f"Loaded GECToR checkpoint from {args.checkpoint}")
    elif os.path.exists(args.bert_checkpoint):
        # Load pretrained BERT
        load_pretrained_bert(args.bert_checkpoint, model)
        print(f"Loaded pretrained BERT from {args.bert_checkpoint}")
    else:
        print("WARNING: No pretrained weights found. Training from scratch.")

    return model


def create_optimizer(model, args, is_frozen):
    """Create optimizer with appropriate parameter groups."""
    if is_frozen:
        # Only train heads
        params = [
            {'params': model.get_head_parameters(), 'lr': args.cold_lr},
        ]
    else:
        # Train everything
        lr_bert = args.lr if args.lr else args.lr_bert
        lr_head = args.lr if args.lr else args.lr_head
        params = [
            {'params': model.get_bert_parameters(), 'lr': lr_bert},
            {'params': model.get_head_parameters(), 'lr': lr_head},
        ]

    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def create_datasets(args):
    """Create train and validation datasets."""
    train_files, val_files, _ = get_file_lists(
        args.data_dir, test_ratio=0.05, val_ratio=0.05, seed=args.seed,
    )

    if args.max_samples:
        train_files = train_files[:args.max_samples]
        val_files = val_files[:min(args.max_samples // 10, len(val_files))]

    include_clean = (args.stage == 3)
    clean_ratio = args.clean_ratio if args.stage == 3 else 0.0

    train_ds = GECToRDataset(
        file_list=train_files,
        data_dir=args.data_dir,
        max_len=args.max_len,
        include_clean=include_clean,
        clean_ratio=clean_ratio,
        p_pitch=args.p_pitch,
        p_rhythm=args.p_rhythm,
        p_delete=args.p_delete,
        p_insert=args.p_insert,
    )
    val_ds = GECToRDataset(
        file_list=val_files,
        data_dir=args.data_dir,
        max_len=args.max_len,
        include_clean=False,
        p_pitch=args.p_pitch,
        p_rhythm=args.p_rhythm,
        p_delete=args.p_delete,
        p_insert=args.p_insert,
    )

    return train_ds, val_ds


def train():
    args = parse_args()

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=_resolve_precision(),
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    set_seed(args.seed)

    is_main = accelerator.is_main_process

    # TensorBoard
    tb_writer = None
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.output_dir, 'tb_logs')
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        except ImportError:
            print("TensorBoard not available, skipping.")

    if is_main:
        print("=" * 60)
        print(f"Music GECToR Training (absolute_bundled) - Stage {args.stage}")
        print("=" * 60)
        print(f"Encoding: absolute_bundled (vocab={VOCAB_SIZE}, labels={NUM_LABELS})")
        print(f"Data: {args.data_dir}")
        print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
        print(f"Freeze epochs: {args.freeze_epochs}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)

    # Data
    train_ds, val_ds = create_datasets(args)
    if is_main:
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    collator = GECToRCollator(max_length=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers if not args.debug else 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers if not args.debug else 0,
        pin_memory=True,
    )

    # Model
    model = build_model(args)

    # Stage I: freeze BERT initially
    is_frozen = False
    if args.stage == 1 and args.freeze_epochs > 0:
        model.freeze_bert()
        is_frozen = True
        if is_main:
            print(f"BERT frozen for first {args.freeze_epochs} epochs")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer & scheduler
    optimizer = create_optimizer(model, args, is_frozen)

    num_update_steps = len(train_loader) // args.gradient_accumulation
    total_steps = num_update_steps * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if is_main:
        print(f"Steps/epoch: {num_update_steps}, Total: {total_steps}, Warmup: {warmup_steps}")

    # Prepare
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(args.epochs):
        # Check if we need to unfreeze BERT
        if is_frozen and epoch >= args.freeze_epochs:
            accelerator.unwrap_model(model).unfreeze_bert()
            is_frozen = False

            # Recreate optimizer with BERT parameters
            optimizer = create_optimizer(accelerator.unwrap_model(model), args, False)
            # Re-prepare optimizer
            optimizer = accelerator.prepare(optimizer)

            # New scheduler for remaining epochs
            remaining_steps = num_update_steps * (args.epochs - epoch)
            remaining_warmup = int(remaining_steps * 0.05)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=remaining_warmup,
                num_training_steps=remaining_steps,
            )
            scheduler = accelerator.prepare(scheduler)

            if is_main:
                trainable = sum(
                    p.numel() for p in accelerator.unwrap_model(model).parameters()
                    if p.requires_grad
                )
                print(f"\n[Epoch {epoch+1}] BERT unfrozen. Trainable: {trainable:,}")

        model.train()
        epoch_tag_loss = 0.0
        epoch_detect_loss = 0.0
        epoch_total_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                detect_logits, tag_logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )

                total_loss, tag_loss, detect_loss = compute_loss(
                    detect_logits, tag_logits,
                    batch['detect_labels'], batch['labels'],
                    keep_weight=args.keep_weight,
                    lambda_detect=args.lambda_detect,
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_total_loss += total_loss.item()
            epoch_tag_loss += tag_loss.item()
            epoch_detect_loss += detect_loss.item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % args.log_every == 0 and is_main:
                    avg_total = epoch_total_loss / epoch_steps
                    avg_tag = epoch_tag_loss / epoch_steps
                    avg_det = epoch_detect_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(
                        f"E{epoch+1} S{global_step} | "
                        f"Loss {avg_total:.4f} (tag={avg_tag:.4f} det={avg_det:.4f}) | "
                        f"LR {lr:.2e} | {elapsed:.0f}s"
                    )
                    if tb_writer:
                        tb_writer.add_scalar('train/total_loss', avg_total, global_step)
                        tb_writer.add_scalar('train/tag_loss', avg_tag, global_step)
                        tb_writer.add_scalar('train/detect_loss', avg_det, global_step)
                        tb_writer.add_scalar('train/lr', lr, global_step)

                # Eval
                if global_step % args.eval_every == 0:
                    val_metrics = evaluate(model, val_loader, accelerator, args)
                    if is_main:
                        print(
                            f"  [Eval] S{global_step} | "
                            f"Loss {val_metrics['total_loss']:.4f} | "
                            f"Tag {val_metrics['tag_loss']:.4f} | "
                            f"Det {val_metrics['detect_loss']:.4f} | "
                            f"EditF1 {val_metrics['edit_f1']:.4f}"
                        )
                        if tb_writer:
                            for k, v in val_metrics.items():
                                tb_writer.add_scalar(f'eval/{k}', v, global_step)

                        if val_metrics['total_loss'] < best_val_loss:
                            best_val_loss = val_metrics['total_loss']
                            patience_counter = 0
                            save_checkpoint(model, accelerator, args, 'best_model')
                            print(f"  [Best] Saved (loss={best_val_loss:.4f})")
                        else:
                            patience_counter += 1

                    model.train()

                # Periodic save
                if global_step % args.save_every == 0 and is_main:
                    save_checkpoint(model, accelerator, args, f'step_{global_step}')

        # End of epoch
        if is_main:
            avg_loss = epoch_total_loss / max(epoch_steps, 1)
            elapsed = time.time() - t0
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | Loss {avg_loss:.4f} | {elapsed/60:.1f}min ---\n")

        # Early stopping
        if patience_counter >= args.patience:
            if is_main:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Final save
    if is_main:
        save_checkpoint(model, accelerator, args, 'final_model')
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        if tb_writer:
            tb_writer.close()


@torch.no_grad()
def evaluate(model, val_loader, accelerator, args, max_batches=50):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_tag_loss = 0.0
    total_detect_loss = 0.0
    num_batches = 0

    # For Edit F1
    tp = 0  # true positive edits
    fp = 0  # false positive edits
    fn = 0  # false negative edits

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        detect_logits, tag_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        loss, tag_loss, detect_loss = compute_loss(
            detect_logits, tag_logits,
            batch['detect_labels'], batch['labels'],
            keep_weight=args.keep_weight,
            lambda_detect=args.lambda_detect,
        )

        total_loss += loss.item()
        total_tag_loss += tag_loss.item()
        total_detect_loss += detect_loss.item()
        num_batches += 1

        # Edit F1 computation
        pred_labels = tag_logits.argmax(dim=-1)  # (batch, seq_len)
        true_labels = batch['labels']
        mask = (true_labels != LABEL_PAD)

        pred_edit = (pred_labels != 0) & mask  # predicted non-KEEP
        true_edit = (true_labels != 0) & mask  # actual non-KEEP

        tp += (pred_edit & true_edit).sum().item()
        fp += (pred_edit & ~true_edit).sum().item()
        fn += (~pred_edit & true_edit).sum().item()

    n = max(num_batches, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'total_loss': total_loss / n,
        'tag_loss': total_tag_loss / n,
        'detect_loss': total_detect_loss / n,
        'edit_f1': f1,
        'edit_precision': precision,
        'edit_recall': recall,
    }


def save_checkpoint(model, accelerator, args, name):
    """Save model checkpoint."""
    save_dir = os.path.join(args.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), os.path.join(save_dir, 'model.pt'))

    # Save args
    import json
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    train()
