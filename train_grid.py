#!/usr/bin/env python3
"""
Study 2 Phase 2: Training grid for scale-sensitivity experiment.
Trains 3 model sizes × 3 training durations × 5 seeds = 45 AR models on AO-CHILDES.

Environment: Python 3.12, torch 2.4.1, torch-directml, transformers 5.3.0
Hardware: AMD RX 7900 GRE (16GB VRAM) via DirectML

Usage:
    python train_grid.py --corpus data/aochildes.txt --out_dir checkpoints/grid
    python train_grid.py --corpus data/aochildes.txt --out_dir checkpoints/grid --resume
    python train_grid.py --corpus data/aochildes.txt --out_dir checkpoints/grid --only_size small
    python train_grid.py --corpus data/aochildes.txt --out_dir checkpoints/grid --only_seed 0

Author: ****
Date: 2026-03-06
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Model architecture specification."""
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    approx_params: str  # for logging

MODEL_SPECS = {
    "small":  ModelSpec("small",  n_layer=4, n_head=4,  n_embd=128, approx_params="~2M"),
    "medium": ModelSpec("medium", n_layer=6, n_head=8,  n_embd=256, approx_params="~7M"),
    "large":  ModelSpec("large",  n_layer=8, n_head=8,  n_embd=512, approx_params="~30M"),
}

EPOCH_COUNTS = [5, 10, 20]
SEEDS = [0, 1, 2, 3, 4]

# Training hyperparameters (matched to pilot: 6L-256d → PPL 36.4)
BATCH_SIZE = 32
BLOCK_SIZE = 128          # context window — AO-CHILDES sentences are short
LEARNING_RATE = 5e-4
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION = 1
MAX_GRAD_NORM = 1.0

# Nonce tokens to add to vocabulary (space-prefixed, per LOCKED decision)
NONCE_TOKENS = [
    " wug", " dax", " blicket", " toma", " zib",
    " mep", " fep", " gorp", " snarp", " pilk",
    " neem", " boff", " spog", " terg", " vun",
    " kib", " chet", " rax", " lub", " poag",
]


# ─────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────

class ChunkedTextDataset(Dataset):
    """Tokenise corpus into fixed-length chunks for causal LM training."""

    def __init__(self, token_ids: list[int], block_size: int):
        self.block_size = block_size
        # Drop last incomplete chunk
        n_chunks = len(token_ids) // (block_size + 1)
        total = n_chunks * (block_size + 1)
        self.data = torch.tensor(token_ids[:total], dtype=torch.long)
        self.data = self.data.view(n_chunks, block_size + 1)
        logging.info(f"Dataset: {n_chunks} chunks of {block_size+1} tokens")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # input, target


# ─────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────

def get_or_train_tokenizer(corpus_path: str, tokenizer_dir: str, vocab_size: int = 8000) -> GPT2TokenizerFast:
    """Load existing tokenizer or train BPE from corpus."""
    tok_path = Path(tokenizer_dir)
    if (tok_path / "tokenizer.json").exists():
        logging.info(f"Loading tokenizer from {tok_path}")
        tokenizer = GPT2TokenizerFast.from_pretrained(str(tok_path))
    else:
        logging.info(f"Training BPE tokenizer (vocab={vocab_size}) on {corpus_path}")
        from tokenizers import ByteLevelBPETokenizer
        bpe = ByteLevelBPETokenizer()
        bpe.train(
            files=[corpus_path],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<pad>"],
        )
        tok_path.mkdir(parents=True, exist_ok=True)
        bpe.save_model(str(tok_path))
        # Save in HF format
        tokenizer = GPT2TokenizerFast.from_pretrained(str(tok_path))
        tokenizer.pad_token = "<pad>"
        tokenizer.save_pretrained(str(tok_path))

    # Add nonce tokens if not already present
    new_tokens = [t for t in NONCE_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        n_added = tokenizer.add_tokens(new_tokens)
        logging.info(f"Added {n_added} nonce tokens to vocabulary")
        tokenizer.save_pretrained(str(tok_path))

    return tokenizer


# ─────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────

def build_model(spec: ModelSpec, vocab_size: int) -> GPT2LMHeadModel:
    """Build GPT-2 model from spec."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=BLOCK_SIZE,
        n_embd=spec.n_embd,
        n_layer=spec.n_layer,
        n_head=spec.n_head,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model {spec.name}: {spec.n_layer}L-{spec.n_embd}d, {n_params:,} params ({spec.approx_params})")
    return model


# ─────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────

def train_one_model(
    spec: ModelSpec,
    n_epochs: int,
    seed: int,
    tokenizer: GPT2TokenizerFast,
    token_ids: list[int],
    out_dir: str,
    device,
) -> dict:
    """Train a single model configuration. Returns training metadata."""

    run_name = f"{spec.name}_ep{n_epochs}_s{seed}"
    run_dir = Path(out_dir) / run_name

    # Check if already completed
    meta_path = run_dir / "training_meta.json"
    if meta_path.exists():
        logging.info(f"SKIP {run_name}: already completed")
        with open(meta_path) as f:
            return json.load(f)

    logging.info(f"\n{'='*60}")
    logging.info(f"TRAINING: {run_name}")
    logging.info(f"  Architecture: {spec.n_layer}L-{spec.n_embd}d ({spec.approx_params})")
    logging.info(f"  Epochs: {n_epochs}, Seed: {seed}")
    logging.info(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    if hasattr(torch, 'cuda'):
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_model(spec, len(tokenizer))
    model = model.to(device)

    # Dataset
    dataset = ChunkedTextDataset(token_ids, BLOCK_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # DirectML requires main-thread data
        generator=torch.Generator().manual_seed(seed),
    )

    # Optimizer + scheduler
    total_steps = len(loader) * n_epochs
    warmup_steps = int(total_steps * WARMUP_FRACTION)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training
    model.train()
    t0 = time.time()
    epoch_losses = []
    step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss, "ppl": ppl})

        # Log every epoch for short runs, every 5 for long
        if n_epochs <= 10 or (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, PPL={ppl:.1f}")

    train_time = time.time() - t0
    final_ppl = epoch_losses[-1]["ppl"]

    logging.info(f"  DONE: PPL={final_ppl:.1f}, time={train_time:.0f}s")

    # Save checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    # Save metadata
    meta = {
        "run_name": run_name,
        "model_spec": asdict(spec),
        "n_epochs": n_epochs,
        "seed": seed,
        "final_loss": epoch_losses[-1]["loss"],
        "final_ppl": final_ppl,
        "train_time_seconds": train_time,
        "total_steps": step,
        "epoch_log": epoch_losses,
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_fraction": WARMUP_FRACTION,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "vocab_size": len(tokenizer),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Free memory
    del model, optimizer, scheduler
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()

    return meta


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Study 2: Training grid (45 AR models)")
    parser.add_argument("--corpus", required=True, help="Path to AO-CHILDES text file")
    parser.add_argument("--out_dir", default="checkpoints/grid", help="Output directory for checkpoints")
    parser.add_argument("--tokenizer_dir", default=None, help="Tokenizer directory (default: out_dir/tokenizer)")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs")
    parser.add_argument("--only_size", choices=["small", "medium", "large"], help="Train only one model size")
    parser.add_argument("--only_epochs", type=int, choices=EPOCH_COUNTS, help="Train only one epoch count")
    parser.add_argument("--only_seed", type=int, choices=SEEDS, help="Train only one seed")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of DirectML")
    parser.add_argument("--vocab_size", type=int, default=8000, help="BPE vocabulary size")
    args = parser.parse_args()

    # Create output directory before logging (FileHandler needs it to exist)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.out_dir) / "train_grid.log", mode="a"),
        ],
    )

    # Device
    if args.cpu:
        device = torch.device("cpu")
    else:
        try:
            device = torch_directml.device()
            logging.info(f"Using DirectML device: {device}")
        except Exception:
            logging.warning("DirectML unavailable, falling back to CPU")
            device = torch.device("cpu")

    # Tokenizer
    tok_dir = args.tokenizer_dir or str(Path(args.out_dir) / "tokenizer")
    tokenizer = get_or_train_tokenizer(args.corpus, tok_dir, args.vocab_size)
    logging.info(f"Vocabulary size: {len(tokenizer)}")

    # Tokenise corpus (once, reuse for all models)
    logging.info("Tokenising corpus...")
    with open(args.corpus, "r") as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    logging.info(f"Corpus: {len(token_ids):,} tokens")

    # Build run grid
    sizes = [args.only_size] if args.only_size else list(MODEL_SPECS.keys())
    epochs = [args.only_epochs] if args.only_epochs else EPOCH_COUNTS
    seeds = [args.only_seed] if args.only_seed is not None else SEEDS

    total_runs = len(sizes) * len(epochs) * len(seeds)
    logging.info(f"\nTraining grid: {len(sizes)} sizes × {len(epochs)} durations × {len(seeds)} seeds = {total_runs} runs")

    # Train all configurations
    results = []
    completed = 0
    skipped = 0
    t_total = time.time()

    for size_name in sizes:
        spec = MODEL_SPECS[size_name]
        for n_ep in epochs:
            for s in seeds:
                run_name = f"{spec.name}_ep{n_ep}_s{s}"
                meta_path = Path(args.out_dir) / run_name / "training_meta.json"

                if args.resume and meta_path.exists():
                    skipped += 1
                    with open(meta_path) as f:
                        results.append(json.load(f))
                    continue

                try:
                    meta = train_one_model(
                        spec=spec,
                        n_epochs=n_ep,
                        seed=s,
                        tokenizer=tokenizer,
                        token_ids=token_ids,
                        out_dir=args.out_dir,
                        device=device,
                    )
                    results.append(meta)
                    completed += 1
                except Exception as e:
                    logging.error(f"FAILED: {run_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    total_time = time.time() - t_total

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"GRID TRAINING COMPLETE")
    logging.info(f"  Completed: {completed}, Skipped: {skipped}, Total: {total_runs}")
    logging.info(f"  Wall time: {total_time/3600:.1f} hours")
    logging.info(f"{'='*60}")

    # Save summary table
    summary_path = Path(args.out_dir) / "grid_summary.json"
    summary = {
        "total_runs": total_runs,
        "completed": completed,
        "skipped": skipped,
        "wall_time_hours": total_time / 3600,
        "results": [
            {
                "run_name": r["run_name"],
                "size": r["model_spec"]["name"],
                "n_layer": r["model_spec"]["n_layer"],
                "n_embd": r["model_spec"]["n_embd"],
                "n_epochs": r["n_epochs"],
                "seed": r["seed"],
                "final_ppl": r["final_ppl"],
                "train_time_s": r["train_time_seconds"],
                "n_params": r["n_params"],
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved: {summary_path}")

    # Print PPL table
    print("\n" + "─" * 70)
    print(f"{'Model':<10} {'Epochs':<8} {'Seed 0':<10} {'Seed 1':<10} {'Seed 2':<10} {'Seed 3':<10} {'Seed 4':<10}")
    print("─" * 70)
    for size_name in sizes:
        for n_ep in epochs:
            ppls = []
            for s in seeds:
                matches = [r for r in results if r["model_spec"]["name"] == size_name
                          and r["n_epochs"] == n_ep and r["seed"] == s]
                if matches:
                    ppls.append(f"{matches[0]['final_ppl']:.1f}")
                else:
                    ppls.append("FAIL")
            print(f"{size_name:<10} {n_ep:<8} {'  '.join(f'{p:<8}' for p in ppls)}")
    print("─" * 70)


if __name__ == "__main__":
    main()
