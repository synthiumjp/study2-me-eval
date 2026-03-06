#!/usr/bin/env python3
"""
Study 2 Phase 2: Batch evaluation of ME battery on all grid models.

Runs three evaluation tracks on each checkpoint:
  1. Familiar-familiar suppression (Finding 2: repetition priming / anti-ME)
  2. Nonce token ME with context-dependence diagnostic (Finding 3)
  3. Repetition priming dose-response (Priority 4: 1, 2, 3 labelling repetitions)

Environment: Python 3.12, torch 2.4.1, torch-directml, transformers 5.3.0
Hardware: AMD RX 7900 GRE (16GB VRAM) via DirectML

Usage:
    python eval_grid.py --grid_dir checkpoints/grid --out results/grid_eval.json
    python eval_grid.py --grid_dir checkpoints/grid --out results/grid_eval.json --only_track suppression
    python eval_grid.py --checkpoint checkpoints/ar_pilot --out results/pilot_eval.json

Author: ++++
Date: 2026-03-06
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch_directml
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np

# ─────────────────────────────────────────────────────
# Nonce token configuration (must match train_grid.py)
# ─────────────────────────────────────────────────────

NONCE_TOKENS = [
    " wug", " dax", " blicket", " toma", " zib",
    " mep", " fep", " gorp", " snarp", " pilk",
    " neem", " boff", " spog", " terg", " vun",
    " kib", " chet", " rax", " lub", " poag",
]

# ─────────────────────────────────────────────────────
# ME battery items
# ─────────────────────────────────────────────────────

# Familiar-familiar suppression pairs (from Phase 1 pilot, Finding 2)
SUPPRESSION_PAIRS = [
    ("ball", "book"),
    ("ball", "car"),
    ("dog", "cat"),
    ("book", "car"),
    ("cup", "hat"),
    ("ball", "dog"),
    ("bird", "fish"),
    ("car", "hat"),
    ("cup", "book"),
    ("dog", "ball"),
]

# Nonce ME items: (familiar, nonce1, nonce2)
# nonce1 = referent for nonce2 to map to; nonce2 = the novel label
NONCE_ME_ITEMS = [
    ("ball", " wug", " dax"),
    ("book", " toma", " zib"),
    ("car", " mep", " fep"),
    ("hat", " gorp", " snarp"),
    ("bird", " pilk", " neem"),
    ("cup", " boff", " spog"),
    ("dog", " terg", " vun"),
    ("cat", " kib", " chet"),
]

# Context-dependence diagnostic conditions
CONTEXT_CONDITIONS = [
    "full_context",    # "there is a [fam] and a [nonce1] . the [nonce2] is the ___"
    "swap_context",    # "there is a [nonce1] and a [fam] . the [nonce2] is the ___"
    "nonce_only",      # "there is a [nonce1] . the [nonce2] is the ___"
    "fam_only",        # "there is a [fam] . the [nonce2] is the ___"
    "no_preamble",     # "the [nonce2] is the ___"
]


# ─────────────────────────────────────────────────────
# Scoring utilities
# ─────────────────────────────────────────────────────

def get_logprob(model, tokenizer, prompt: str, target: str, device) -> float:
    """Get log probability of target token given prompt (teacher-forced)."""
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Get target token id (must be single token)
    target_ids = tokenizer.encode(target)
    # Handle space-prefix: " ball" → single token; "ball" might need space
    if len(target_ids) == 0:
        return float("-inf")
    # Take last token if tokenizer adds special tokens
    target_id = target_ids[-1] if len(target_ids) > 1 else target_ids[0]

    with torch.no_grad():
        outputs = model(input_ids=prompt_ids)
        logits = outputs.logits[0, -1, :]  # last position
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[target_id].item()


def verify_single_token(tokenizer, word: str) -> bool:
    """Verify a word tokenises to exactly one token."""
    ids = tokenizer.encode(word)
    return len(ids) == 1


# ─────────────────────────────────────────────────────
# Track 1: Familiar-familiar suppression
# ─────────────────────────────────────────────────────

def eval_suppression(model, tokenizer, device) -> list[dict]:
    """
    Test familiar-familiar suppression (anti-ME / repetition priming).

    Baseline:    "there is a [fam1] and a [fam2] . this is a"
    ME condition: "there is a [fam1] and a [fam2] . that is a [fam1] . this is a"

    ME prediction: P(fam2|ME) > P(fam2|base) and P(fam1|ME) < P(fam1|base)
    Anti-ME:      P(fam1|ME) > P(fam1|base) (repetition priming)
    """
    results = []

    for fam1, fam2 in SUPPRESSION_PAIRS:
        # Both directions
        for target_first, target_second in [(fam1, fam2), (fam2, fam1)]:
            base_prompt = f"there is a {target_first} and a {target_second} . this is a"
            me_prompt = f"there is a {target_first} and a {target_second} . that is a {target_first} . this is a"

            lp_fam1_base = get_logprob(model, tokenizer, base_prompt, f" {target_first}", device)
            lp_fam2_base = get_logprob(model, tokenizer, base_prompt, f" {target_second}", device)
            lp_fam1_me = get_logprob(model, tokenizer, me_prompt, f" {target_first}", device)
            lp_fam2_me = get_logprob(model, tokenizer, me_prompt, f" {target_second}", device)

            # Suppression = base - me (positive = suppressed in ME condition)
            fam1_suppression = lp_fam1_base - lp_fam1_me  # negative = PRIMED (anti-ME)
            fam2_boost = lp_fam2_me - lp_fam2_base  # positive = boosted (ME-consistent)

            results.append({
                "pair": f"{target_first}/{target_second}",
                "labelled": target_first,
                "unlabelled": target_second,
                "lp_labelled_base": lp_fam1_base,
                "lp_labelled_me": lp_fam1_me,
                "lp_unlabelled_base": lp_fam2_base,
                "lp_unlabelled_me": lp_fam2_me,
                "labelled_suppression": fam1_suppression,
                "unlabelled_boost": fam2_boost,
                "is_anti_me": fam1_suppression < 0,  # priming, not suppression
                "is_me_consistent": fam2_boost > 0 and fam1_suppression > 0,
            })

    return results


# ─────────────────────────────────────────────────────
# Track 2: Nonce ME + context-dependence diagnostic
# ─────────────────────────────────────────────────────

def build_nonce_prompt(condition: str, fam: str, nonce1: str, nonce2: str) -> str:
    """Build prompt for a given context-dependence condition."""
    if condition == "full_context":
        return f"there is a{fam} and a{nonce1} . the{nonce2} is the"
    elif condition == "swap_context":
        return f"there is a{nonce1} and a{fam} . the{nonce2} is the"
    elif condition == "nonce_only":
        return f"there is a{nonce1} . the{nonce2} is the"
    elif condition == "fam_only":
        return f"there is a{fam} . the{nonce2} is the"
    elif condition == "no_preamble":
        return f"the{nonce2} is the"
    else:
        raise ValueError(f"Unknown condition: {condition}")


def init_nonce_embeddings(model, tokenizer, seed: int):
    """
    Initialise nonce token embeddings using Strategy B:
    random rare-noun anchor + 10% noise per nonce.
    """
    rng = np.random.RandomState(seed)
    embed_weight = model.get_input_embeddings().weight.data

    # Find rare single-token nouns to use as anchors
    # These are tokens with IDs in the vocabulary that are real words
    vocab = tokenizer.get_vocab()
    all_ids = list(range(len(vocab)))

    for nonce in NONCE_TOKENS:
        if nonce not in vocab:
            continue
        nonce_id = vocab[nonce]

        # Pick random anchor from existing vocabulary
        anchor_id = rng.choice(all_ids)
        anchor_emb = embed_weight[anchor_id].clone()

        # Add 10% Gaussian noise
        noise = torch.randn_like(anchor_emb) * 0.1 * anchor_emb.abs().mean()
        embed_weight[nonce_id] = anchor_emb + noise

    # Also update output projection (tied weights)
    if hasattr(model, 'lm_head') and model.lm_head.weight is not model.get_input_embeddings().weight:
        model.lm_head.weight.data = model.get_input_embeddings().weight.data.clone()


def eval_nonce_me(model, tokenizer, device, n_embed_seeds: int = 10) -> dict:
    """
    Evaluate nonce ME items with context-dependence diagnostic.
    Tests across multiple embedding initialisation seeds.
    """
    all_results = []

    for embed_seed in range(n_embed_seeds):
        # Re-init nonce embeddings for this seed
        init_nonce_embeddings(model, tokenizer, embed_seed)
        model.eval()

        seed_results = []
        for fam, nonce1, nonce2 in NONCE_ME_ITEMS:
            item_results = {"familiar": fam, "nonce1": nonce1.strip(), "nonce2": nonce2.strip(), "embed_seed": embed_seed}

            for condition in CONTEXT_CONDITIONS:
                prompt = build_nonce_prompt(condition, f" {fam}", nonce1, nonce2)

                lp_nonce1 = get_logprob(model, tokenizer, prompt, nonce1, device)
                lp_fam = get_logprob(model, tokenizer, prompt, f" {fam}", device)

                item_results[f"{condition}_lp_nonce"] = lp_nonce1
                item_results[f"{condition}_lp_fam"] = lp_fam
                item_results[f"{condition}_diff"] = lp_nonce1 - lp_fam  # positive = ME-consistent
                item_results[f"{condition}_me"] = lp_nonce1 > lp_fam

            seed_results.append(item_results)

        all_results.append(seed_results)

    # Aggregate across embedding seeds
    summary = {}
    for condition in CONTEXT_CONDITIONS:
        me_counts = []
        diffs = []
        for seed_results in all_results:
            n_me = sum(1 for item in seed_results if item[f"{condition}_me"])
            mean_diff = np.mean([item[f"{condition}_diff"] for item in seed_results])
            me_counts.append(n_me)
            diffs.append(mean_diff)

        summary[condition] = {
            "mean_me_count": np.mean(me_counts),
            "std_me_count": np.std(me_counts),
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "me_counts_by_seed": me_counts,
        }

    return {
        "item_results": all_results,
        "summary": summary,
        "n_embed_seeds": n_embed_seeds,
        "n_items": len(NONCE_ME_ITEMS),
    }


# ─────────────────────────────────────────────────────
# Track 3: Repetition priming dose-response
# ─────────────────────────────────────────────────────

def eval_priming_dose(model, tokenizer, device) -> list[dict]:
    """
    Measure repetition priming as a function of labelling repetitions.

    0 reps (baseline): "there is a [fam1] and a [fam2] . this is a"
    1 rep:             "there is a [fam1] and a [fam2] . that is a [fam1] . this is a"
    2 reps:            "... that is a [fam1] . see the [fam1] . this is a"
    3 reps:            "... that is a [fam1] . see the [fam1] . get the [fam1] . this is a"

    Prediction: priming increases monotonically (opposite of ME which would saturate/reverse).
    """
    results = []

    # Use subset of pairs for dose-response
    dose_pairs = [("ball", "book"), ("dog", "cat"), ("cup", "hat"), ("car", "bird"), ("book", "dog")]

    for fam1, fam2 in dose_pairs:
        base = f"there is a {fam1} and a {fam2}"
        reps = [
            f"{base} . this is a",                                                                    # 0 reps
            f"{base} . that is a {fam1} . this is a",                                                 # 1 rep
            f"{base} . that is a {fam1} . see the {fam1} . this is a",                                # 2 reps
            f"{base} . that is a {fam1} . see the {fam1} . get the {fam1} . this is a",               # 3 reps
        ]

        pair_results = {"pair": f"{fam1}/{fam2}", "doses": []}

        for n_reps, prompt in enumerate(reps):
            lp_fam1 = get_logprob(model, tokenizer, prompt, f" {fam1}", device)
            lp_fam2 = get_logprob(model, tokenizer, prompt, f" {fam2}", device)

            pair_results["doses"].append({
                "n_reps": n_reps,
                "lp_labelled": lp_fam1,
                "lp_unlabelled": lp_fam2,
                "diff": lp_fam1 - lp_fam2,  # positive = repetition priming
            })

        # Check monotonicity: is priming increasing with dose?
        diffs = [d["diff"] for d in pair_results["doses"]]
        pair_results["monotonic_increase"] = all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1))
        pair_results["priming_slope"] = (diffs[-1] - diffs[0]) / 3  # avg increase per rep

        results.append(pair_results)

    return results


# ─────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device):
    """Load model and tokenizer from checkpoint."""
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint_dir)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────

def eval_one_checkpoint(checkpoint_dir: str, device, tracks: list[str]) -> dict:
    """Run specified evaluation tracks on one checkpoint."""
    logging.info(f"  Loading: {checkpoint_dir}")
    model, tokenizer = load_model(checkpoint_dir, device)

    results = {"checkpoint": checkpoint_dir}

    if "suppression" in tracks:
        logging.info("    Running: familiar-familiar suppression")
        results["suppression"] = eval_suppression(model, tokenizer, device)

        # Quick summary
        n_anti = sum(1 for r in results["suppression"] if r["is_anti_me"])
        n_total = len(results["suppression"])
        mean_prime = np.mean([r["labelled_suppression"] for r in results["suppression"]])
        logging.info(f"    → Anti-ME: {n_anti}/{n_total}, mean priming: {mean_prime:.2f} nats")

    if "nonce" in tracks:
        logging.info("    Running: nonce ME + context diagnostic")
        results["nonce_me"] = eval_nonce_me(model, tokenizer, device, n_embed_seeds=10)

        # Quick summary
        for cond in CONTEXT_CONDITIONS:
            s = results["nonce_me"]["summary"][cond]
            logging.info(f"    → {cond}: ME={s['mean_me_count']:.1f}/8, diff={s['mean_diff']:.2f}")

    if "dose" in tracks:
        logging.info("    Running: repetition priming dose-response")
        results["priming_dose"] = eval_priming_dose(model, tokenizer, device)

        n_mono = sum(1 for r in results["priming_dose"] if r["monotonic_increase"])
        logging.info(f"    → Monotonic: {n_mono}/{len(results['priming_dose'])}")

    # Free memory
    del model
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Study 2: Batch ME evaluation on grid models")
    parser.add_argument("--grid_dir", help="Directory containing grid checkpoints (from train_grid.py)")
    parser.add_argument("--checkpoint", help="Evaluate a single checkpoint")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--tracks", nargs="+", default=["suppression", "nonce", "dose"],
                        choices=["suppression", "nonce", "dose"],
                        help="Which evaluation tracks to run")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of DirectML")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Device
    if args.cpu:
        device = torch.device("cpu")
    else:
        try:
            device = torch_directml.device()
            logging.info(f"Using DirectML device: {device}")
        except Exception:
            device = torch.device("cpu")

    # Collect checkpoints
    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    elif args.grid_dir:
        grid_path = Path(args.grid_dir)
        for run_dir in sorted(grid_path.iterdir()):
            if run_dir.is_dir() and (run_dir / "config.json").exists():
                checkpoints.append(str(run_dir))
        logging.info(f"Found {len(checkpoints)} checkpoints in {args.grid_dir}")
    else:
        parser.error("Must specify --grid_dir or --checkpoint")

    # Run evaluations
    all_results = []
    t0 = time.time()

    for i, ckpt in enumerate(checkpoints):
        logging.info(f"\n[{i+1}/{len(checkpoints)}] Evaluating: {Path(ckpt).name}")
        try:
            result = eval_one_checkpoint(ckpt, device, args.tracks)

            # Load training metadata if available
            meta_path = Path(ckpt) / "training_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    result["training_meta"] = json.load(f)

            all_results.append(result)
        except Exception as e:
            logging.error(f"FAILED: {ckpt}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - t0

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "n_checkpoints": len(all_results),
        "tracks": args.tracks,
        "eval_time_seconds": total_time,
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

    logging.info(f"\nResults saved: {out_path}")
    logging.info(f"Total eval time: {total_time/60:.1f} min")

    # ─────────────────────────────────────────────────
    # Print summary tables
    # ─────────────────────────────────────────────────

    if "suppression" in args.tracks:
        print("\n" + "=" * 70)
        print("FAMILIAR-FAMILIAR SUPPRESSION (anti-ME rate)")
        print("=" * 70)
        print(f"{'Checkpoint':<30} {'Anti-ME':<12} {'Mean priming':<15}")
        print("-" * 70)
        for r in all_results:
            name = Path(r["checkpoint"]).name
            supp = r.get("suppression", [])
            n_anti = sum(1 for s in supp if s["is_anti_me"])
            mean_p = np.mean([s["labelled_suppression"] for s in supp]) if supp else 0
            print(f"{name:<30} {n_anti}/{len(supp):<10} {mean_p:>+.2f} nats")

    if "nonce" in args.tracks:
        print("\n" + "=" * 70)
        print("CONTEXT-DEPENDENCE DIAGNOSTIC (nonce ME)")
        print("=" * 70)
        print(f"{'Checkpoint':<20} ", end="")
        for cond in CONTEXT_CONDITIONS:
            print(f"{cond[:12]:<14}", end="")
        print()
        print("-" * 90)
        for r in all_results:
            name = Path(r["checkpoint"]).name[:19]
            nonce = r.get("nonce_me", {}).get("summary", {})
            print(f"{name:<20} ", end="")
            for cond in CONTEXT_CONDITIONS:
                if cond in nonce:
                    print(f"{nonce[cond]['mean_me_count']:.1f}/8       ", end="")
                else:
                    print(f"{'N/A':<14}", end="")
            print()

    if "dose" in args.tracks:
        print("\n" + "=" * 70)
        print("REPETITION PRIMING DOSE-RESPONSE")
        print("=" * 70)
        print(f"{'Checkpoint':<30} {'Monotonic':<12} {'Slope (nats/rep)':<18}")
        print("-" * 70)
        for r in all_results:
            name = Path(r["checkpoint"]).name
            dose = r.get("priming_dose", [])
            n_mono = sum(1 for d in dose if d["monotonic_increase"])
            mean_slope = np.mean([d["priming_slope"] for d in dose]) if dose else 0
            print(f"{name:<30} {n_mono}/{len(dose):<10} {mean_slope:>+.3f}")


if __name__ == "__main__":
    main()
