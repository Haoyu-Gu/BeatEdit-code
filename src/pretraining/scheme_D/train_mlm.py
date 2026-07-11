"""
Music BERT MLM pretraining script (absolute_bundled encoding scheme - Scheme D).

Uses HuggingFace BertForMaskedLM + Accelerate for distributed training.
"""
import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

from config import MusicTokenConfig, BertPretrainConfig
from mlm_dataset import MLMDataset, BucketBatchSampler, MLMCollator


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(token_config: MusicTokenConfig, bert_config: BertPretrainConfig):
    """Build the BertForMaskedLM model."""
    config = BertConfig(
        vocab_size=token_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        max_position_embeddings=bert_config.max_position_embeddings,
        hidden_dropout_prob=bert_config.hidden_dropout_prob,
        attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
        pad_token_id=token_config.pad_token_id,
        type_vocab_size=1,  # no segment embedding needed
    )
    model = BertForMaskedLM(config)
    return model


def train():
    parser = argparse.ArgumentParser(description='Music BERT MLM Pre-training (absolute_bundled)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='directory of preprocessed .npz files '
                             '(overrides BEATEDIT_DATA_DIR and the config default)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--grad_accum', type=int, default=None, help='gradient accumulation steps')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--tb_log_dir', type=str, default=None, help='tensorboard log directory')
    args = parser.parse_args()

    # Configuration
    token_config = MusicTokenConfig()
    bert_config = BertPretrainConfig()

    if args.data_dir is not None:
        bert_config.data_dir = args.data_dir
    if args.epochs is not None:
        bert_config.num_epochs = args.epochs
    if args.batch_size is not None:
        bert_config.batch_size = args.batch_size
    if args.lr is not None:
        bert_config.learning_rate = args.lr
    if args.output_dir is not None:
        bert_config.output_dir = args.output_dir
    if args.grad_accum is not None:
        bert_config.gradient_accumulation_steps = args.grad_accum

    # Accelerator
    # fp16 needs CUDA; fall back to full precision on CPU/MPS so a small
    # pilot run works anywhere (override with BEATEDIT_PRECISION).
    if bert_config.mixed_precision == "fp16" and not torch.cuda.is_available():
        bert_config.mixed_precision = "no"

    accelerator = Accelerator(
        mixed_precision=bert_config.mixed_precision,
        gradient_accumulation_steps=bert_config.gradient_accumulation_steps,
    )
    set_seed(bert_config.random_seed)

    # Tensorboard
    tb_writer = None
    if accelerator.is_main_process:
        tb_log_dir = args.tb_log_dir or os.path.join(bert_config.output_dir, 'tb_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"Tensorboard log dir: {tb_log_dir}")

    if accelerator.is_main_process:
        os.makedirs(bert_config.output_dir, exist_ok=True)
        print("=" * 60)
        print("Music BERT MLM Pre-training (absolute_bundled)")
        print("=" * 60)
        print(f"Encoding: absolute_bundled (vocab={token_config.vocab_size})")
        print(f"Model: hidden={bert_config.hidden_size}, layers={bert_config.num_hidden_layers}, "
              f"heads={bert_config.num_attention_heads}")
        print(f"Data: {bert_config.data_dir}")
        print(f"Batch: {bert_config.batch_size} x {bert_config.gradient_accumulation_steps} accum")
        print(f"LR: {bert_config.learning_rate}, Epochs: {bert_config.num_epochs}")
        print(f"Output: {bert_config.output_dir}")
        print("=" * 60)

    # Datasets
    train_dataset = MLMDataset(
        data_dir=bert_config.data_dir,
        token_config=token_config,
        bert_config=bert_config,
        mode='train',
    )
    eval_dataset = MLMDataset(
        data_dir=bert_config.data_dir,
        token_config=token_config,
        bert_config=bert_config,
        mode='test',
    )

    collator = MLMCollator(
        pad_token_id=token_config.pad_token_id,
        max_length=bert_config.max_seq_len,
    )

    # DataLoader
    if train_dataset.sorted_indices is not None:
        train_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=bert_config.batch_size,
            bucket_size=bert_config.batch_size * 10,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=bert_config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=bert_config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=bert_config.num_workers,
            pin_memory=True,
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=bert_config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=bert_config.num_workers,
        pin_memory=True,
    )

    # Model
    model = build_model(token_config, bert_config)

    total_params, trainable_params = count_parameters(model)
    if accelerator.is_main_process:
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': bert_config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=bert_config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Learning-rate schedule
    num_update_steps_per_epoch = len(train_loader) // bert_config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * bert_config.num_epochs
    warmup_steps = int(total_training_steps * bert_config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if accelerator.is_main_process:
        print(f"Steps/epoch: {num_update_steps_per_epoch}, Total: {total_training_steps}, Warmup: {warmup_steps}")

    # Accelerate prepare
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        accelerator.load_state(args.resume)
        # Parse the step from the checkpoint directory name
        ckpt_name = os.path.basename(args.resume)
        if 'step_' in ckpt_name:
            global_step = int(ckpt_name.split('step_')[1])
            start_epoch = global_step // num_update_steps_per_epoch
        if accelerator.is_main_process:
            print(f"Resumed from {args.resume}, step={global_step}, epoch={start_epoch}")

    # Training loop
    if accelerator.is_main_process:
        print("\nStarting training...")

    best_eval_loss = float('inf')

    for epoch in range(start_epoch, bert_config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        step_in_epoch = 0
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

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Statistics
            batch_masked = (batch['labels'] != -100).sum().item()
            epoch_loss += loss.item() * batch_masked
            epoch_tokens += batch_masked
            step_in_epoch += 1

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % bert_config.log_every_n_steps == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    tokens_per_sec = epoch_tokens / elapsed
                    print(
                        f"Epoch {epoch+1}/{bert_config.num_epochs} | "
                        f"Step {global_step} | "
                        f"Loss {avg_loss:.4f} | "
                        f"PPL {math.exp(min(avg_loss, 10)):.2f} | "
                        f"LR {lr:.2e} | "
                        f"Tok/s {tokens_per_sec:.0f}"
                    )
                    # Tensorboard
                    if tb_writer:
                        tb_writer.add_scalar('train/loss', avg_loss, global_step)
                        tb_writer.add_scalar('train/ppl', math.exp(min(avg_loss, 10)), global_step)
                        tb_writer.add_scalar('train/lr', lr, global_step)
                        tb_writer.add_scalar('train/tokens_per_sec', tokens_per_sec, global_step)

                # Evaluation
                if global_step % bert_config.eval_every_n_steps == 0:
                    eval_loss = evaluate(model, eval_loader, accelerator)
                    if accelerator.is_main_process:
                        eval_ppl = math.exp(min(eval_loss, 10))
                        print(
                            f"  [Eval] Step {global_step} | "
                            f"Loss {eval_loss:.4f} | "
                            f"PPL {eval_ppl:.2f}"
                        )
                        if tb_writer:
                            tb_writer.add_scalar('eval/loss', eval_loss, global_step)
                            tb_writer.add_scalar('eval/ppl', eval_ppl, global_step)
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_path = os.path.join(bert_config.output_dir, 'best_model')
                            accelerator.save_state(save_path)
                            print(f"  [Best] saved to {save_path}")
                    model.train()

                # Periodic saving
                if global_step % bert_config.save_every_n_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(bert_config.output_dir, f'step_{global_step}')
                    accelerator.save_state(save_path)
                    print(f"  [Save] {save_path}")

        # End of epoch
        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(epoch_tokens, 1)
            elapsed = time.time() - t0
            print(
                f"\n--- Epoch {epoch+1} done | "
                f"Avg Loss {avg_loss:.4f} | "
                f"PPL {math.exp(min(avg_loss, 10)):.2f} | "
                f"Time {elapsed/60:.1f} min ---\n"
            )
            if tb_writer:
                tb_writer.add_scalar('epoch/train_loss', avg_loss, epoch + 1)

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(bert_config.output_dir, 'final_model')
        accelerator.save_state(save_path)
        print(f"Training complete! Final model saved to {save_path}")
        print(f"Best eval loss: {best_eval_loss:.4f}")
        if tb_writer:
            tb_writer.close()


@torch.no_grad()
def evaluate(model, eval_loader, accelerator, max_batches=50):
    """Evaluate MLM loss on the test set."""
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

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


if __name__ == '__main__':
    train()
