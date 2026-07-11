"""
REMI Music BERT MLM Pretraining

Uses HuggingFace BertForMaskedLM + Accelerate for distributed training.
Masks music content tokens (Pitch, Velocity, Duration, Position), NOT structural tokens.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 --main_process_port=29501 \
        train_mlm.py --epochs 30 --batch_size 64 --output_dir checkpoints/music_bert_remi
"""

import os
import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import VOCAB_SIZE, PAD_TOKEN, MIDI_DATA_DIR
from dataset import REMIMLMDataset, MLMCollator, BucketBatchSampler, get_file_lists
from model import get_bert_config


# fp16 needs CUDA; fall back to full precision elsewhere so a small pilot run
# works on CPU/MPS. Override with BEATEDIT_PRECISION=no|fp16|bf16.
def _resolve_precision(configured='fp16'):
    requested = os.environ.get('BEATEDIT_PRECISION', configured)
    if requested in ('fp16', 'bf16') and not torch.cuda.is_available():
        return 'no'
    return requested


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model():
    """Build BertForMaskedLM with REMI config."""
    config = get_bert_config()
    model = BertForMaskedLM(config)
    return model


def train():
    parser = argparse.ArgumentParser(description='REMI Music BERT MLM Pre-training')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/music_bert_remi')
    parser.add_argument('--data_dir', type=str, default=MIDI_DATA_DIR)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

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
            pass

    if is_main:
        print("=" * 60)
        print("REMI Music BERT MLM Pre-training")
        print("=" * 60)
        print(f"Encoding: REMI (vocab={VOCAB_SIZE})")
        print(f"Model: hidden=512, layers=8, heads=8")
        print(f"Data: {args.data_dir}")
        print(f"Batch: {args.batch_size} x {args.gradient_accumulation} accum")
        print(f"LR: {args.lr}, Epochs: {args.epochs}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)

    # Data
    train_files, val_files, test_files = get_file_lists(args.data_dir, seed=args.seed)
    if args.max_samples:
        train_files = train_files[:args.max_samples]
        test_files = test_files[:min(args.max_samples // 10, len(test_files))]

    train_dataset = REMIMLMDataset(
        file_list=train_files,
        data_dir=args.data_dir,
        max_len=args.max_len,
    )
    eval_dataset = REMIMLMDataset(
        file_list=test_files,
        data_dir=args.data_dir,
        max_len=args.max_len,
    )

    collator = MLMCollator(pad_token_id=PAD_TOKEN, max_length=args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if is_main:
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Model
    model = build_model()
    total_params, trainable_params = count_parameters(model)
    if is_main:
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation
    total_training_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if is_main:
        print(f"Steps/epoch: {num_update_steps_per_epoch}, Total: {total_training_steps}, Warmup: {warmup_steps}")

    # Prepare
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        accelerator.load_state(args.resume)
        ckpt_name = os.path.basename(args.resume)
        if 'step_' in ckpt_name:
            global_step = int(ckpt_name.split('step_')[1])
            start_epoch = global_step // num_update_steps_per_epoch
        if is_main:
            print(f"Resumed from {args.resume}, step={global_step}, epoch={start_epoch}")

    # Training
    if is_main:
        print("\nStarting training...")

    best_eval_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            batch_masked = (batch['labels'] != -100).sum().item()
            epoch_loss += loss.item() * batch_masked
            epoch_tokens += batch_masked

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_every == 0 and is_main:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    tokens_per_sec = epoch_tokens / elapsed
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step} | "
                        f"Loss {avg_loss:.4f} | "
                        f"PPL {math.exp(min(avg_loss, 10)):.2f} | "
                        f"LR {lr:.2e} | "
                        f"Tok/s {tokens_per_sec:.0f}"
                    )
                    if tb_writer:
                        tb_writer.add_scalar('train/loss', avg_loss, global_step)
                        tb_writer.add_scalar('train/ppl', math.exp(min(avg_loss, 10)), global_step)
                        tb_writer.add_scalar('train/lr', lr, global_step)

                if global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, accelerator)
                    if is_main:
                        eval_ppl = math.exp(min(eval_loss, 10))
                        print(f"  [Eval] Step {global_step} | Loss {eval_loss:.4f} | PPL {eval_ppl:.2f}")
                        if tb_writer:
                            tb_writer.add_scalar('eval/loss', eval_loss, global_step)
                            tb_writer.add_scalar('eval/ppl', eval_ppl, global_step)
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_path = os.path.join(args.output_dir, 'best_model')
                            accelerator.save_state(save_path)
                            print(f"  [Best] Saved to {save_path}")
                    model.train()

                if global_step % args.save_every == 0 and is_main:
                    save_path = os.path.join(args.output_dir, f'step_{global_step}')
                    accelerator.save_state(save_path)
                    print(f"  [Save] {save_path}")

        if is_main:
            avg_loss = epoch_loss / max(epoch_tokens, 1)
            elapsed = time.time() - t0
            print(f"\n--- Epoch {epoch+1} done | Avg Loss {avg_loss:.4f} | PPL {math.exp(min(avg_loss, 10)):.2f} | {elapsed/60:.1f}min ---\n")

    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, 'final_model')
        accelerator.save_state(save_path)
        print(f"Training complete! Final model saved to {save_path}")
        print(f"Best eval loss: {best_eval_loss:.4f}")
        if tb_writer:
            tb_writer.close()


@torch.no_grad()
def evaluate(model, eval_loader, accelerator, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        masked = (batch['labels'] != -100).sum().item()
        total_loss += outputs.loss.item() * masked
        total_tokens += masked

    return total_loss / max(total_tokens, 1)


if __name__ == '__main__':
    train()
