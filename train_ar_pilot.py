#!/usr/bin/env python3
"""
train_ar_pilot.py — Train a small GPT-2 on AO-CHILDES for ME pilot.

Usage:
    python train_ar_pilot.py --data data/aochildes.txt --output checkpoints/ar_pilot

Trains a 6-layer, 256-dim GPT-2 (~7M params) on AO-CHILDES.
Expects ~30-60 min on AMD RX 7900 GRE via DirectML.
Falls back to CPU if DirectML unavailable.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


class LineDataset(Dataset):
    """Tokenizes corpus lines and creates fixed-length chunks."""

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        # Truncate to exact multiple of seq_len
        n = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # input, target


def train_tokenizer(corpus_path, vocab_size=8192, output_dir=None):
    """Train a BPE tokenizer on the corpus."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>"],
        min_frequency=2,
    )

    tokenizer.train([corpus_path], trainer)

    # Add post-processor for byte-level
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = tokenizers_decoder_bytelevel()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    return tokenizer


def tokenizers_decoder_bytelevel():
    from tokenizers import decoders
    return decoders.ByteLevel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/aochildes.txt")
    parser.add_argument("--output", default="checkpoints/ar_pilot")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_every_epoch", action="store_true", default=False)
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────
    tok_path = os.path.join(args.output, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[INFO] Loading existing tokenizer from {tok_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        print(f"[INFO] Training BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = train_tokenizer(args.data, args.vocab_size, args.output)
        print(f"[INFO] Tokenizer saved to {tok_path}")

    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    actual_vocab = tokenizer.get_vocab_size()
    print(f"[INFO] Vocab size: {actual_vocab}, pad_id={pad_id}, eos_id={eos_id}")

    # Verify key words are single-token
    test_words = ["ball", "dog", "cup", "cat", "hat", "book", "car", "asp", "helm", "rib", "cot", "gob", "cork"]
    print("[INFO] Token check:")
    for w in test_words:
        enc = tokenizer.encode(f" {w}")
        print(f"  {w:10s} -> {len(enc.ids)} tokens: {enc.ids} {enc.tokens}")

    # ── Tokenize corpus ──────────────────────────────────────────────────
    print(f"[INFO] Tokenizing corpus...")
    all_ids = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                enc = tokenizer.encode(line)
                all_ids.extend(enc.ids)
                all_ids.append(eos_id)

    print(f"[INFO] Total tokens: {len(all_ids):,}")

    dataset = LineDataset(all_ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[INFO] {len(dataset)} chunks of length {args.seq_len}, {len(loader)} batches/epoch")

    # ── Model ────────────────────────────────────────────────────────────
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=actual_vocab,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.seq_len,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {args.n_layer}L-{args.n_embd}d, {n_params/1e6:.1f}M params")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Training loop ────────────────────────────────────────────────────
    print(f"[INFO] Training for {args.epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            outputs = model(x, labels=y)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % args.log_every == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step+1}/{len(loader)} | loss {avg:.4f} | ppl {math.exp(avg):.1f} | {elapsed:.0f}s")

        avg_loss = total_loss / n_batches
        ppl = math.exp(avg_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | loss {avg_loss:.4f} | ppl {ppl:.1f} | {elapsed:.0f}s elapsed")

        if args.save_every_epoch:
            ckpt = os.path.join(args.output, f"epoch_{epoch}")
            model.to("cpu")
            model.save_pretrained(ckpt)
            model.to(device)
            print(f"  Saved checkpoint to {ckpt}")

    # ── Save final ───────────────────────────────────────────────────────
    model.to("cpu")
    model.save_pretrained(args.output)

    # Save tokenizer in HuggingFace format for compatibility with scoring pipeline
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
    )
    hf_tokenizer.save_pretrained(args.output)

    # Save training config
    train_config = vars(args)
    train_config["n_params"] = n_params
    train_config["total_tokens"] = len(all_ids)
    train_config["final_loss"] = avg_loss
    train_config["final_ppl"] = ppl
    train_config["training_time_s"] = time.time() - t0
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\n[DONE] Model saved to {args.output}")
    print(f"  Final loss: {avg_loss:.4f}, PPL: {ppl:.1f}")
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
