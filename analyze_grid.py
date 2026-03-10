#!/usr/bin/env python3
"""
Study 2 Phase 2: Pre-registered statistical analyses on grid evaluation results.

Computes:
  CONFIRMATORY:
    H1: Sign test on suppression scores per cell
    H2: Wilcoxon signed-rank on nonce_only vs full_context per cell
    H3: Kendall's tau-b + bootstrap CI on dose-response slope per cell
    H4: Conjunction of H1 across all cells

  EXPLORATORY:
    - Correlation: model size × priming magnitude
    - Correlation: training duration × priming magnitude
    - Size × duration interaction
    - PPL × priming relationship
    - Per-item stability analysis

Usage:
    python analyze_grid.py --eval results/grid_eval.json --grid_summary checkpoints/grid/grid_summary.json --out results/analysis_report.txt

Author: JP Cacioli
Date: 2026-03-08
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import product

# ─────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────

def load_data(eval_path, summary_path=None):
    """Load evaluation results and optional training summary."""
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Build lookup: run_name → result
    results = {}
    for r in eval_data["results"]:
        # Extract run name from checkpoint path
        name = Path(r["checkpoint"]).name
        results[name] = r

        # Attach training metadata if embedded
        if "training_meta" in r:
            results[name]["ppl"] = r["training_meta"]["final_ppl"]

    # If separate summary provided, merge PPL data
    if summary_path:
        with open(summary_path) as f:
            summary = json.load(f)
        for s in summary["results"]:
            if s["run_name"] in results:
                results[s["run_name"]]["ppl"] = s["final_ppl"]
                results[s["run_name"]]["n_params"] = s["n_params"]

    return results


def get_cell_models(results, size, epochs):
    """Get all models (across seeds) for a given size × epochs cell."""
    cell = []
    for name, r in results.items():
        if f"{size}_ep{epochs}_" in name:
            cell.append(r)
    return cell


SIZES = ["small", "medium", "large"]
EPOCH_COUNTS = [5, 10, 20]
SEEDS = [0, 1, 2, 3, 4]
CONTEXT_CONDITIONS = ["full_context", "swap_context", "nonce_only", "fam_only", "no_preamble"]


# ─────────────────────────────────────────────────────
# H1: Sign test on suppression scores
# ─────────────────────────────────────────────────────

def analyze_h1(results, out):
    """H1: Anti-ME in all cells. Sign test on suppression scores."""
    out.append("=" * 70)
    out.append("H1: FAMILIAR-FAMILIAR SUPPRESSION — SIGN TEST PER CELL")
    out.append("Pre-registered: anti-ME rate > 50% in ALL 9 cells")
    out.append("Sign test: one-tailed, H0: median suppression = 0")
    out.append("=" * 70)
    out.append("")
    out.append(f"{'Cell':<20} {'Anti-ME':<12} {'Mean (nats)':<14} {'Sign test p':<14} {'Confirmed'}")
    out.append("-" * 70)

    all_confirmed = True
    cell_results = {}

    for size in SIZES:
        for ep in EPOCH_COUNTS:
            cell_name = f"{size}_ep{ep}"
            models = get_cell_models(results, size, ep)

            # Collect all suppression scores across seeds
            all_scores = []
            n_anti = 0
            n_total = 0
            for m in models:
                for item in m.get("suppression", []):
                    score = item["labelled_suppression"]
                    all_scores.append(score)
                    if score < 0:  # anti-ME (priming)
                        n_anti += 1
                    n_total += 1

            anti_rate = n_anti / n_total if n_total > 0 else 0
            mean_score = np.mean(all_scores)

            # One-tailed sign test: H0 median >= 0, H1 median < 0
            n_negative = sum(1 for s in all_scores if s < 0)
            n_nonzero = sum(1 for s in all_scores if s != 0)
            # Binomial test: probability of observing this many negatives if p=0.5
            p_val = stats.binomtest(n_negative, n_nonzero, 0.5, alternative='greater').pvalue

            confirmed = anti_rate > 0.5
            if not confirmed:
                all_confirmed = False

            cell_results[cell_name] = {
                "anti_rate": anti_rate,
                "mean_score": mean_score,
                "p_value": p_val,
                "n_anti": n_anti,
                "n_total": n_total,
                "confirmed": confirmed,
            }

            out.append(f"{cell_name:<20} {n_anti}/{n_total:<10} {mean_score:>+.2f}        {p_val:<14.2e} {'YES' if confirmed else 'NO'}")

    out.append("")
    out.append(f"H1 OVERALL: {'CONFIRMED — anti-ME in all 9 cells' if all_confirmed else 'DISCONFIRMED'}")
    out.append("")
    return cell_results, all_confirmed


# ─────────────────────────────────────────────────────
# H2: Wilcoxon on nonce_only vs full_context
# ─────────────────────────────────────────────────────

def analyze_h2(results, out):
    """H2: Context-dependence diagnostic replicates. nonce_only >= full_context."""
    out.append("=" * 70)
    out.append("H2: CONTEXT-DEPENDENCE DIAGNOSTIC — WILCOXON PER CELL")
    out.append("Pre-registered: nonce_only >= full_context in ALL 9 cells")
    out.append("=" * 70)
    out.append("")
    out.append(f"{'Cell':<20} {'nonce_only':<12} {'full_ctx':<12} {'Wilcoxon p':<14} {'Confirmed'}")
    out.append("-" * 70)

    all_confirmed = True

    for size in SIZES:
        for ep in EPOCH_COUNTS:
            cell_name = f"{size}_ep{ep}"
            models = get_cell_models(results, size, ep)

            nonce_only_counts = []
            full_context_counts = []

            for m in models:
                nonce_me = m.get("nonce_me", {})
                summary = nonce_me.get("summary", {})
                if "nonce_only" in summary and "full_context" in summary:
                    nonce_only_counts.append(summary["nonce_only"]["mean_me_count"])
                    full_context_counts.append(summary["full_context"]["mean_me_count"])

            mean_nonce = np.mean(nonce_only_counts) if nonce_only_counts else 0
            mean_full = np.mean(full_context_counts) if full_context_counts else 0

            # Wilcoxon signed-rank: paired across seeds
            if len(nonce_only_counts) >= 5:
                diffs = [n - f for n, f in zip(nonce_only_counts, full_context_counts)]
                try:
                    stat, p_val = stats.wilcoxon(diffs, alternative='greater')
                except ValueError:
                    p_val = 1.0  # all zeros
            else:
                p_val = float('nan')

            confirmed = mean_nonce >= mean_full
            if not confirmed:
                all_confirmed = False

            out.append(f"{cell_name:<20} {mean_nonce:.1f}/8      {mean_full:.1f}/8      {p_val:<14.4f} {'YES' if confirmed else 'NO'}")

    out.append("")
    out.append(f"H2 OVERALL: {'CONFIRMED — nonce_only >= full_context in all 9 cells' if all_confirmed else 'DISCONFIRMED'}")
    out.append("")
    return all_confirmed


# ─────────────────────────────────────────────────────
# H3: Kendall's tau-b + bootstrap CI on slope
# ─────────────────────────────────────────────────────

def analyze_h3(results, out):
    """H3: Monotonic priming dose-response. Kendall's tau-b + slope CI."""
    out.append("=" * 70)
    out.append("H3: REPETITION PRIMING DOSE-RESPONSE — KENDALL TAU-B PER CELL")
    out.append("Pre-registered: >50% pairs monotonic in ALL 9 cells")
    out.append("=" * 70)
    out.append("")
    out.append(f"{'Cell':<20} {'Monotonic':<12} {'Mean slope':<14} {'Kendall tau':<14} {'tau p-val':<14} {'Confirmed'}")
    out.append("-" * 70)

    all_confirmed = True

    for size in SIZES:
        for ep in EPOCH_COUNTS:
            cell_name = f"{size}_ep{ep}"
            models = get_cell_models(results, size, ep)

            all_slopes = []
            n_mono = 0
            n_total = 0

            # Collect all dose-response data for Kendall's tau
            all_doses = []  # (dose_level, priming_diff) pairs

            for m in models:
                for pair in m.get("priming_dose", []):
                    if pair.get("monotonic_increase"):
                        n_mono += 1
                    n_total += 1
                    all_slopes.append(pair.get("priming_slope", 0))

                    for dose_item in pair.get("doses", []):
                        all_doses.append((dose_item["n_reps"], dose_item["diff"]))

            mono_rate = n_mono / n_total if n_total > 0 else 0
            mean_slope = np.mean(all_slopes) if all_slopes else 0

            # Kendall's tau-b across all dose observations
            if all_doses:
                doses_x = [d[0] for d in all_doses]
                doses_y = [d[1] for d in all_doses]
                tau, tau_p = stats.kendalltau(doses_x, doses_y)
            else:
                tau, tau_p = 0, 1

            # Bootstrap CI on slope
            if all_slopes:
                boot_slopes = []
                rng = np.random.RandomState(42)
                for _ in range(10000):
                    sample = rng.choice(all_slopes, size=len(all_slopes), replace=True)
                    boot_slopes.append(np.mean(sample))
                ci_lo = np.percentile(boot_slopes, 2.5)
                ci_hi = np.percentile(boot_slopes, 97.5)
            else:
                ci_lo, ci_hi = 0, 0

            confirmed = mono_rate > 0.5
            if not confirmed:
                all_confirmed = False

            out.append(f"{cell_name:<20} {n_mono}/{n_total:<10} {mean_slope:>+.3f}       {tau:>+.3f}         {tau_p:<14.2e} {'YES' if confirmed else 'NO'}")
            out.append(f"{'':20} Slope 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    out.append("")
    out.append(f"H3 OVERALL: {'CONFIRMED — monotonic priming in all 9 cells' if all_confirmed else 'PARTIALLY CONFIRMED (see details)'}")
    out.append("")
    return all_confirmed


# ─────────────────────────────────────────────────────
# H4: Conjunction
# ─────────────────────────────────────────────────────

def analyze_h4(h1_confirmed, out):
    """H4: No ME at any scale. Conjunction of H1."""
    out.append("=" * 70)
    out.append("H4: NO ME AT ANY SCALE — CONJUNCTION OF H1")
    out.append("=" * 70)
    out.append("")
    out.append(f"H4: {'CONFIRMED — no cell shows ME-consistent suppression' if h1_confirmed else 'DISCONFIRMED'}")
    out.append("")


# ─────────────────────────────────────────────────────
# Exploratory analyses
# ─────────────────────────────────────────────────────

def analyze_exploratory(results, out):
    """Exploratory: correlations, interactions, per-item analysis."""
    out.append("=" * 70)
    out.append("EXPLORATORY ANALYSES (not pre-registered)")
    out.append("=" * 70)
    out.append("")

    # Collect per-model summaries
    model_data = []
    for name, r in results.items():
        if "suppression" not in r:
            continue

        # Parse size and epochs from name
        parts = name.split("_")
        size_name = parts[0]
        size_map = {"small": 2862848, "medium": 8878080, "large": 33498112}
        n_params = r.get("n_params", size_map.get(size_name, 0))

        ep_str = parts[1]  # "ep5", "ep10", "ep20"
        n_epochs = int(ep_str.replace("ep", ""))

        mean_priming = np.mean([item["labelled_suppression"] for item in r["suppression"]])
        ppl = r.get("ppl", 0)

        model_data.append({
            "name": name,
            "size": size_name,
            "n_params": n_params,
            "n_epochs": n_epochs,
            "mean_priming": mean_priming,
            "ppl": ppl,
        })

    if not model_data:
        out.append("No data available for exploratory analyses.")
        return

    params = [m["n_params"] for m in model_data]
    epochs = [m["n_epochs"] for m in model_data]
    priming = [m["mean_priming"] for m in model_data]
    ppls = [m["ppl"] for m in model_data]

    # 1. Model size × priming magnitude
    r_size, p_size = stats.spearmanr(params, priming)
    out.append(f"1. Model size vs priming magnitude:")
    out.append(f"   Spearman r = {r_size:+.3f}, p = {p_size:.4f}")
    out.append(f"   (Positive r = larger models show weaker priming / closer to zero)")
    out.append("")

    # 2. Training duration × priming magnitude
    r_ep, p_ep = stats.spearmanr(epochs, priming)
    out.append(f"2. Training duration vs priming magnitude:")
    out.append(f"   Spearman r = {r_ep:+.3f}, p = {p_ep:.4f}")
    out.append(f"   (Positive r = more training shows weaker priming / closer to zero)")
    out.append("")

    # 3. PPL × priming
    if any(p > 0 for p in ppls):
        r_ppl, p_ppl = stats.spearmanr(ppls, priming)
        out.append(f"3. Perplexity vs priming magnitude:")
        out.append(f"   Spearman r = {r_ppl:+.3f}, p = {p_ppl:.4f}")
        out.append(f"   (Negative r = better LMs (lower PPL) show weaker priming)")
        out.append("")

    # 4. Cell-level summary table
    out.append("4. Mean priming (nats) by size × epochs:")
    out.append(f"   {'':15} {'5 ep':>10} {'10 ep':>10} {'20 ep':>10}")
    out.append("   " + "-" * 45)
    for size in SIZES:
        row = f"   {size:<15}"
        for ep in EPOCH_COUNTS:
            cell_models = [m for m in model_data if m["size"] == size and m["n_epochs"] == ep]
            if cell_models:
                mean = np.mean([m["mean_priming"] for m in cell_models])
                sd = np.std([m["mean_priming"] for m in cell_models])
                row += f" {mean:>+.2f}±{sd:.2f}"
            else:
                row += f" {'N/A':>10}"
        out.append(row)
    out.append("")

    # 5. PPL table
    out.append("5. Mean PPL by size × epochs:")
    out.append(f"   {'':15} {'5 ep':>10} {'10 ep':>10} {'20 ep':>10}")
    out.append("   " + "-" * 45)
    for size in SIZES:
        row = f"   {size:<15}"
        for ep in EPOCH_COUNTS:
            cell_models = [m for m in model_data if m["size"] == size and m["n_epochs"] == ep]
            if cell_models:
                mean_ppl = np.mean([m["ppl"] for m in cell_models])
                row += f" {mean_ppl:>9.1f}"
            else:
                row += f" {'N/A':>10}"
        out.append(row)
    out.append("")

    # 6. Per-item stability: which noun pairs show strongest priming?
    out.append("6. Per-item priming strength (mean across all 45 models):")
    item_scores = {}
    for name, r in results.items():
        for item in r.get("suppression", []):
            pair = item["pair"]
            if pair not in item_scores:
                item_scores[pair] = []
            item_scores[pair].append(item["labelled_suppression"])

    out.append(f"   {'Pair':<20} {'Mean (nats)':<14} {'SD':<10} {'Anti-ME %'}")
    out.append("   " + "-" * 55)
    for pair in sorted(item_scores.keys(), key=lambda p: np.mean(item_scores[p])):
        scores = item_scores[pair]
        mean_s = np.mean(scores)
        sd_s = np.std(scores)
        anti_pct = 100 * sum(1 for s in scores if s < 0) / len(scores)
        out.append(f"   {pair:<20} {mean_s:>+.3f}        {sd_s:.3f}     {anti_pct:.0f}%")
    out.append("")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Study 2: Pre-registered analysis of grid evaluation")
    parser.add_argument("--eval", required=True, help="Path to grid_eval.json")
    parser.add_argument("--grid_summary", help="Path to grid_summary.json (for PPL data)")
    parser.add_argument("--out", default="results/analysis_report.txt", help="Output report file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    results = load_data(args.eval, args.grid_summary)
    logging.info(f"Loaded {len(results)} model results")

    out = []
    out.append("=" * 70)
    out.append("STUDY 2: PRE-REGISTERED ANALYSIS REPORT")
    out.append("Scale Sensitivity of Referential Mechanisms in Child-Scale LMs")
    out.append("=" * 70)
    out.append("")

    # Confirmatory
    h1_results, h1_confirmed = analyze_h1(results, out)
    h2_confirmed = analyze_h2(results, out)
    h3_confirmed = analyze_h3(results, out)
    analyze_h4(h1_confirmed, out)

    # Summary
    out.append("=" * 70)
    out.append("CONFIRMATORY SUMMARY")
    out.append("=" * 70)
    out.append(f"  H1 (anti-ME all cells):          {'CONFIRMED' if h1_confirmed else 'DISCONFIRMED'}")
    out.append(f"  H2 (diagnostic replicates):       {'CONFIRMED' if h2_confirmed else 'DISCONFIRMED'}")
    out.append(f"  H3 (monotonic priming):           {'CONFIRMED' if h3_confirmed else 'PARTIALLY CONFIRMED'}")
    out.append(f"  H4 (no ME at any scale):          {'CONFIRMED' if h1_confirmed else 'DISCONFIRMED'}")
    out.append("")

    # Exploratory
    analyze_exploratory(results, out)

    # Write report
    report = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)

    # Also print to console
    print(report)

    logging.info(f"Report saved: {args.out}")


if __name__ == "__main__":
    main()
