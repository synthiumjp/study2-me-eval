#!/usr/bin/env python3
"""
me_scoring_pipeline.py — Study 2 ME Evaluation Scoring Pipeline

Takes a model (AR or masked LM) and the ME item battery JSON,
extracts log-probabilities, computes ME metrics, and outputs results.

Usage:
    # AR model (GPT-2 style, with or without nonce tokens):
    python me_scoring_pipeline.py \
        --model_path ./checkpoints/ar_model \
        --model_type ar \
        --items me_item_battery.json \
        --output results_ar.json

    # BabyBERTa (masked LM):
    python me_scoring_pipeline.py \
        --model_path ./checkpoints/babyberta \
        --model_type mlm \
        --items me_item_battery.json \
        --output results_bb.json

    # Add nonce tokens to AR model vocabulary before scoring:
    python me_scoring_pipeline.py \
        --model_path ./checkpoints/ar_model \
        --model_type ar \
        --items me_item_battery.json \
        --add_nonce_tokens wug dax blicket toma zib mep fep gorp \
        --output results_ar_nonce.json

Environment:
    Requires: torch, transformers, numpy
    Tested: Python 3.10+, PyTorch 2.x (ROCm or CUDA)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─── Model loading ──────────────────────────────────────────────────────────

def load_ar_model(model_path: str, nonce_tokens: Optional[list] = None, device: str = "cpu"):
    """Load an autoregressive (causal LM) model and tokenizer.
    
    If nonce_tokens is provided, adds them to the vocabulary with random
    embeddings (drawn from the existing embedding distribution).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if nonce_tokens:
        # Add nonce tokens to vocabulary
        num_added = tokenizer.add_tokens(nonce_tokens)
        print(f"[INFO] Added {num_added} nonce tokens to vocabulary: {nonce_tokens}")

        # Resize embeddings — new rows get random init
        old_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))

        # Initialise new embeddings from existing distribution
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            existing_mean = emb[:old_size].mean(dim=0)
            existing_std = emb[:old_size].std(dim=0)
            for i in range(old_size, len(tokenizer)):
                emb[i] = torch.normal(existing_mean, existing_std)

        # Verify each nonce token is single-token
        for tok in nonce_tokens:
            ids = tokenizer.encode(f" {tok}", add_special_tokens=False)
            if len(ids) != 1:
                print(f"[WARNING] Nonce token '{tok}' tokenises to {len(ids)} tokens: {ids}")
                print(f"          Expected 1 token. Check tokenizer.add_tokens worked.")
            else:
                print(f"  ✓ '{tok}' → token_id {ids[0]}")

    model.to(device)
    model.eval()
    return model, tokenizer


def load_mlm_model(model_path: str, device: str = "cpu"):
    """Load a masked language model (BabyBERTa / RoBERTa)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


# ─── Log-probability extraction ─────────────────────────────────────────────

def get_ar_logprob(model, tokenizer, prompt: str, target_word: str, device: str = "cpu") -> float:
    """Get log P(target_word | prompt) from an AR model.
    
    Returns log probability of target_word as the next token after prompt.
    For multi-token targets, returns the sum of conditional log probs
    (i.e., log P(t1|ctx) + log P(t2|ctx,t1) + ...).
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    # Tokenize target (with leading space for BPE)
    # Try with space prefix first (standard for GPT-2 BPE)
    target_ids = tokenizer.encode(f" {target_word}", add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)

    n_target = len(target_ids)

    # Concatenate prompt + target for teacher-forced scoring
    full_ids = torch.cat([prompt_ids, torch.tensor([target_ids], device=device)], dim=1)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)

    # Sum log probs of each target token conditioned on everything before it
    total_logprob = 0.0
    prompt_len = prompt_ids.shape[1]
    for i, tid in enumerate(target_ids):
        # Position prompt_len + i - 1 predicts token at position prompt_len + i
        # But for i=0, we use position prompt_len - 1 (last prompt token predicts first target token)
        pos = prompt_len - 1 + i
        total_logprob += log_probs[0, pos, tid].item()

    return total_logprob


def get_mlm_logprob(model, tokenizer, prompt: str, target_word: str, device: str = "cpu") -> float:
    """Get log P(target_word | context with [MASK]) from a masked LM.
    
    The prompt must contain exactly one [MASK] token.
    Returns log probability of target_word at the [MASK] position.
    
    For single-token targets: straightforward.
    For multi-token targets: NOT SUPPORTED (by design — this is the tokenisation
    constraint that motivates the two-track design).
    """
    # Verify target is single-token
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if len(target_ids) != 1:
        print(f"[WARNING] MLM target '{target_word}' is {len(target_ids)} tokens. "
              f"MLM scoring requires single-token targets. Returning -inf.")
        return float('-inf')

    target_id = target_ids[0]

    # Tokenize with special tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Find [MASK] position
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)

    if len(mask_positions[1]) == 0:
        print(f"[ERROR] No [MASK] token found in prompt: {prompt}")
        return float('-inf')
    if len(mask_positions[1]) > 1:
        print(f"[WARNING] Multiple [MASK] tokens found. Using first one.")

    mask_idx = mask_positions[1][0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0, mask_idx, target_id].item()


# ─── Item scoring functions ──────────────────────────────────────────────────

def score_ar_item(model, tokenizer, item: dict, device: str = "cpu") -> dict:
    """Score a single AR item (nonce or pseudo-nonce track)."""
    prompt = item["prompt"]
    result = {"id": item["id"], "track": item["track"], "family": item["family"],
              "condition": item["condition"], "prompt": prompt}

    if item["condition"] in ("me_test", "recency_control"):
        target_me = item.get("target_me") or item.get("pseudo_referent", "")
        target_null = item.get("target_null") or item.get("familiar", "")

        lp_me = get_ar_logprob(model, tokenizer, prompt, target_me, device)
        lp_null = get_ar_logprob(model, tokenizer, prompt, target_null, device)

        result["logp_me_target"] = lp_me
        result["logp_null_target"] = lp_null
        result["me_target_word"] = target_me
        result["null_target_word"] = target_null
        result["logp_diff"] = lp_me - lp_null  # positive = ME direction
        result["me_consistent"] = lp_me > lp_null

    elif item["condition"] in ("nonce_nonce_baseline", "pseudo_pseudo_baseline"):
        n1 = item["noun1"]
        n2 = item["noun2"]
        lp1 = get_ar_logprob(model, tokenizer, prompt, n1, device)
        lp2 = get_ar_logprob(model, tokenizer, prompt, n2, device)
        result["logp_noun1"] = lp1
        result["logp_noun2"] = lp2
        result["noun1"] = n1
        result["noun2"] = n2
        result["logp_diff"] = abs(lp1 - lp2)  # should be ~0 for baselines
        result["bias_direction"] = "noun1" if lp1 > lp2 else "noun2"

    elif item["condition"] == "familiar_familiar_baseline":
        prompt = item.get("prompt_ar", item.get("prompt", ""))
        n1 = item["noun1"]
        n2 = item["noun2"]
        lp1 = get_ar_logprob(model, tokenizer, prompt, n1, device)
        lp2 = get_ar_logprob(model, tokenizer, prompt, n2, device)
        result["prompt"] = prompt
        result["logp_noun1"] = lp1
        result["logp_noun2"] = lp2
        result["noun1"] = n1
        result["noun2"] = n2
        result["logp_diff"] = abs(lp1 - lp2)
        result["bias_direction"] = "noun1" if lp1 > lp2 else "noun2"

    return result


def score_bb_suppression_item(model, tokenizer, item: dict, device: str = "cpu") -> dict:
    """Score a BabyBERTa suppression item.
    
    Compares P(target | ME_context) vs P(target | baseline_context).
    """
    prompt_baseline = item["prompt_baseline"]
    prompt_me = item["prompt_me"]
    fam1 = item["target_fam1"]
    fam2 = item["target_fam2"]

    result = {"id": item["id"], "track": "bb_suppression", "family": item["family"],
              "condition": item["condition"]}

    # Baseline condition
    lp_fam1_base = get_mlm_logprob(model, tokenizer, prompt_baseline, fam1, device)
    lp_fam2_base = get_mlm_logprob(model, tokenizer, prompt_baseline, fam2, device)

    # ME condition
    lp_fam1_me = get_mlm_logprob(model, tokenizer, prompt_me, fam1, device)
    lp_fam2_me = get_mlm_logprob(model, tokenizer, prompt_me, fam2, device)

    result["prompt_baseline"] = prompt_baseline
    result["prompt_me"] = prompt_me
    result["fam1"] = fam1
    result["fam2"] = fam2

    # Baseline probs
    result["logp_fam1_baseline"] = lp_fam1_base
    result["logp_fam2_baseline"] = lp_fam2_base

    # ME probs
    result["logp_fam1_me"] = lp_fam1_me
    result["logp_fam2_me"] = lp_fam2_me

    # Key ME metrics:
    # Suppression: P(fam1) should DROP after fam1 is labelled
    result["fam1_suppression"] = lp_fam1_me - lp_fam1_base  # should be negative
    # Boost: P(fam2) should RISE after fam1 is labelled
    result["fam2_boost"] = lp_fam2_me - lp_fam2_base  # should be positive

    # ME-consistent = fam1 drops AND fam2 rises
    result["me_consistent"] = (lp_fam1_me < lp_fam1_base) and (lp_fam2_me > lp_fam2_base)

    # Composite ME score: boost + |suppression|
    result["me_composite"] = result["fam2_boost"] - result["fam1_suppression"]

    return result


def score_bb_pseudo_item(model, tokenizer, item: dict, device: str = "cpu") -> dict:
    """Score a BabyBERTa pseudo-nonce item (P(pseudo) vs P(familiar) at [MASK])."""
    prompt = item["prompt"]
    target_me = item["target_me"]
    target_null = item["target_null"]

    lp_me = get_mlm_logprob(model, tokenizer, prompt, target_me, device)
    lp_null = get_mlm_logprob(model, tokenizer, prompt, target_null, device)

    result = {"id": item["id"], "track": "bb_pseudo", "family": item["family"],
              "condition": item["condition"], "prompt": prompt,
              "me_target_word": target_me, "null_target_word": target_null,
              "logp_me_target": lp_me, "logp_null_target": lp_null,
              "logp_diff": lp_me - lp_null,
              "me_consistent": lp_me > lp_null}

    return result


# ─── Batch scoring + aggregation ────────────────────────────────────────────

def score_battery(model, tokenizer, items: list, model_type: str, device: str = "cpu") -> list:
    """Score all items in the battery appropriate to model_type."""
    results = []

    for item in items:
        if not isinstance(item, dict):
            continue  # skip comment strings

        track = item.get("track", "")
        condition = item.get("condition", "")

        # Route to correct scorer based on track and model type
        if model_type == "ar":
            if track in ("ar_nonce", "ar_pseudo"):
                results.append(score_ar_item(model, tokenizer, item, device))
            elif track == "all" and condition == "familiar_familiar_baseline":
                results.append(score_ar_item(model, tokenizer, item, device))
            elif track == "all_pseudo" and condition == "pseudo_pseudo_baseline":
                # Use AR prompt
                item_copy = dict(item)
                item_copy["prompt"] = item["prompt_ar"]
                item_copy["condition"] = "pseudo_pseudo_baseline"
                results.append(score_ar_item(model, tokenizer, item_copy, device))

        elif model_type == "mlm":
            if track == "bb_suppression":
                results.append(score_bb_suppression_item(model, tokenizer, item, device))
            elif track == "bb_pseudo":
                results.append(score_bb_pseudo_item(model, tokenizer, item, device))
            elif track == "all" and condition == "familiar_familiar_baseline":
                # Use BB prompt with [MASK]
                item_copy = dict(item)
                item_copy["prompt"] = item["prompt_bb"]
                item_copy["target_me"] = item["noun1"]
                item_copy["target_null"] = item["noun2"]
                results.append(score_bb_pseudo_item(model, tokenizer, item_copy, device))
            elif track == "all_pseudo" and condition == "pseudo_pseudo_baseline":
                item_copy = dict(item)
                item_copy["prompt"] = item["prompt_bb"]
                item_copy["target_me"] = item["noun1"]
                item_copy["target_null"] = item["noun2"]
                results.append(score_bb_pseudo_item(model, tokenizer, item_copy, device))

    return results


def aggregate_results(results: list) -> dict:
    """Compute summary statistics from scored items."""
    summary = {}

    # Group by track × condition
    groups = {}
    for r in results:
        key = (r.get("track", "?"), r.get("condition", "?"))
        groups.setdefault(key, []).append(r)

    for (track, condition), items in sorted(groups.items()):
        group_key = f"{track}__{condition}"
        n = len(items)

        if condition in ("me_test", "recency_control") and track != "bb_suppression":
            diffs = [r["logp_diff"] for r in items if "logp_diff" in r]
            me_count = sum(1 for r in items if r.get("me_consistent", False))
            summary[group_key] = {
                "n": n,
                "me_consistent_count": me_count,
                "me_consistent_pct": me_count / n if n > 0 else 0,
                "mean_logp_diff": float(np.mean(diffs)) if diffs else None,
                "std_logp_diff": float(np.std(diffs)) if diffs else None,
                "min_logp_diff": float(np.min(diffs)) if diffs else None,
                "max_logp_diff": float(np.max(diffs)) if diffs else None,
            }

        elif track == "bb_suppression" and condition in ("me_test", "recency_control",
                                                          "synonym_density_1label",
                                                          "synonym_density_2label"):
            boosts = [r["fam2_boost"] for r in items if "fam2_boost" in r]
            suppressions = [r["fam1_suppression"] for r in items if "fam1_suppression" in r]
            composites = [r["me_composite"] for r in items if "me_composite" in r]
            me_count = sum(1 for r in items if r.get("me_consistent", False))

            summary[group_key] = {
                "n": n,
                "me_consistent_count": me_count,
                "me_consistent_pct": me_count / n if n > 0 else 0,
                "mean_fam2_boost": float(np.mean(boosts)) if boosts else None,
                "mean_fam1_suppression": float(np.mean(suppressions)) if suppressions else None,
                "mean_me_composite": float(np.mean(composites)) if composites else None,
                "std_me_composite": float(np.std(composites)) if composites else None,
            }

        elif "baseline" in condition:
            diffs = [r.get("logp_diff", 0) for r in items]
            summary[group_key] = {
                "n": n,
                "mean_abs_logp_diff": float(np.mean(diffs)) if diffs else None,
                "std_abs_logp_diff": float(np.std(diffs)) if diffs else None,
                "notes": "Should be ~0 for well-controlled baselines"
            }

    return summary


def compute_recency_diagnostic(results: list) -> dict:
    """Compare ME test vs recency-swapped items to detect recency heuristic.
    
    For each item pair (me_test + recency_control):
    - If both ME-consistent: genuine ME signal
    - If me_test consistent but swap flips: recency heuristic
    - If neither consistent: no signal
    """
    # Index by controls_for field
    me_items = {r["id"]: r for r in results if r.get("condition") == "me_test"}
    swap_items = {}
    for r in results:
        if r.get("condition") == "recency_control":
            parent_id = r.get("controls_for", "")
            if parent_id:
                swap_items[parent_id] = r

    diagnostic = {"genuine_me": 0, "recency_heuristic": 0, "no_signal": 0, "total_pairs": 0}

    for me_id, me_r in me_items.items():
        if me_id not in swap_items:
            continue
        swap_r = swap_items[me_id]
        diagnostic["total_pairs"] += 1

        me_ok = me_r.get("me_consistent", False)
        swap_ok = swap_r.get("me_consistent", False)

        if me_ok and swap_ok:
            diagnostic["genuine_me"] += 1
        elif me_ok and not swap_ok:
            diagnostic["recency_heuristic"] += 1
        else:
            diagnostic["no_signal"] += 1

    if diagnostic["total_pairs"] > 0:
        diagnostic["genuine_me_pct"] = diagnostic["genuine_me"] / diagnostic["total_pairs"]
        diagnostic["recency_heuristic_pct"] = diagnostic["recency_heuristic"] / diagnostic["total_pairs"]
    else:
        diagnostic["genuine_me_pct"] = 0
        diagnostic["recency_heuristic_pct"] = 0

    return diagnostic


def compute_synonym_density_comparison(results: list) -> dict:
    """Compare ME strength for 1-label vs 2-label synonym conditions.
    
    Prediction (Byers-Heinlein): ME weaker for 2-label kinds.
    """
    label_1 = [r for r in results if r.get("condition") == "synonym_density_1label"]
    label_2 = [r for r in results if r.get("condition") == "synonym_density_2label"]

    comparison = {}
    if label_1:
        composites_1 = [r["me_composite"] for r in label_1 if "me_composite" in r]
        comparison["1_label"] = {
            "n": len(label_1),
            "mean_me_composite": float(np.mean(composites_1)) if composites_1 else None,
            "std_me_composite": float(np.std(composites_1)) if composites_1 else None,
        }
    if label_2:
        composites_2 = [r["me_composite"] for r in label_2 if "me_composite" in r]
        comparison["2_label"] = {
            "n": len(label_2),
            "mean_me_composite": float(np.mean(composites_2)) if composites_2 else None,
            "std_me_composite": float(np.std(composites_2)) if composites_2 else None,
        }

    if comparison.get("1_label", {}).get("mean_me_composite") is not None and \
       comparison.get("2_label", {}).get("mean_me_composite") is not None:
        diff = comparison["1_label"]["mean_me_composite"] - comparison["2_label"]["mean_me_composite"]
        comparison["density_effect"] = diff
        comparison["density_direction"] = "predicted" if diff > 0 else "unpredicted"

    return comparison


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ME Evaluation Scoring Pipeline — Study 2")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", required=True, choices=["ar", "mlm"],
                        help="Model type: 'ar' for autoregressive, 'mlm' for masked LM")
    parser.add_argument("--items", required=True, help="Path to ME item battery JSON")
    parser.add_argument("--output", required=True, help="Path for output results JSON")
    parser.add_argument("--add_nonce_tokens", nargs="*", default=None,
                        help="Nonce tokens to add to AR model vocabulary")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect cuda/cpu)")
    args = parser.parse_args()

    # Device detection (supports CUDA, DirectML, CPU)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"[INFO] DirectML detected")
        except ImportError:
            device = "cpu"
    print(f"[INFO] Using device: {device}")

    # Load items
    with open(args.items) as f:
        battery = json.load(f)
    items = battery["items"]
    print(f"[INFO] Loaded {sum(1 for i in items if isinstance(i, dict))} items from {args.items}")

    # Load model
    if args.model_type == "ar":
        model, tokenizer = load_ar_model(args.model_path, args.add_nonce_tokens, device)
        print(f"[INFO] Loaded AR model from {args.model_path} (vocab size: {len(tokenizer)})")
    else:
        model, tokenizer = load_mlm_model(args.model_path, device)
        print(f"[INFO] Loaded MLM model from {args.model_path} (vocab size: {len(tokenizer)})")

    # Score
    print("[INFO] Scoring items...")
    results = score_battery(model, tokenizer, items, args.model_type, device)
    print(f"[INFO] Scored {len(results)} items")

    # Aggregate
    summary = aggregate_results(results)
    recency = compute_recency_diagnostic(results)
    synonym = compute_synonym_density_comparison(results) if args.model_type == "mlm" else {}

    # Output
    output = {
        "metadata": {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "device": device,
            "nonce_tokens_added": args.add_nonce_tokens,
            "n_items_scored": len(results),
        },
        "item_results": results,
        "summary": summary,
        "recency_diagnostic": recency,
        "synonym_density_comparison": synonym,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for group_key, stats in sorted(summary.items()):
        print(f"\n--- {group_key} ---")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print(f"\n--- Recency Diagnostic ---")
    for k, v in recency.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if synonym:
        print(f"\n--- Synonym Density (Byers-Heinlein) ---")
        for k, v in synonym.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv:.4f}" if isinstance(vv, float) else f"    {kk}: {vv}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print(f"\n[INFO] Full results saved to {args.output}")


if __name__ == "__main__":
    main()
