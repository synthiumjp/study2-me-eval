# Repetition Without Exclusivity

**Scale Sensitivity of Referential Mechanisms in Child-Scale Language Models**

JP Cacioli · Independent Researcher · Melbourne, Australia

**Pre-registration:** [OSF](https://osf.io/zu7af)
**Paper:** [arXiv](#) *(link to be added after upload)*

---

## Overview

This repository contains code, data, and evaluation results for a study testing whether mutual exclusivity (ME) — the bias to map novel words to novel referents — emerges in text-only language models trained on child-directed speech (AO-CHILDES).

We train 45 GPT-2-architecture models across 3 sizes (2.9M, 8.9M, 33.5M parameters), 3 training durations (5, 10, 20 epochs), and 5 random seeds, and evaluate them on a pre-registered ME battery. We find robust repetition priming (anti-ME) at all scales, with priming attenuating but never reversing as model quality improves.

## Key Findings

- **No ME at any scale:** All 45 models show anti-ME repetition priming (85–100% of items, all *p* < 2.4 × 10⁻¹³).
- **Priming attenuates with quality:** Spearman ρ = −0.533 between perplexity and priming magnitude, but priming never crosses zero.
- **Nonce "ME" is an artefact:** A context-dependence diagnostic shows that apparent ME with nonce tokens is driven by embedding similarity, not referential disambiguation.
- **BabyBERTa is insensitive:** Masked LM predictions are completely invariant to multi-sentence discourse context.

## Repository Structure

```
study2-me-eval/
├── data/
│   └── aochildes.txt              # AO-CHILDES corpus (~4M tokens)
├── checkpoints/
│   ├── grid/                      # 45 trained model checkpoints
│   │   └── grid_summary.json      # PPL and training metadata
│   ├── babyberta/                 # BabyBERTa checkpoint (safetensors)
│   └── ar_pilot/                  # Pilot AR model
├── results/
│   ├── grid_eval.json             # Full evaluation results (3 tracks × 45 models)
│   └── analysis_report.txt        # Pre-registered statistical analysis
├── train_grid.py                  # Training script for 45-model grid
├── eval_grid.py                   # Evaluation script (suppression + nonce + dose)
├── analyze_grid.py                # Pre-registered confirmatory + exploratory statistics
├── corpus_analysis_script.py      # CDS repetition statistics
├── corpus_noun_stats.json         # Output of corpus analysis
├── me_item_battery.json           # 106-item ME evaluation battery
├── me_scoring_pipeline.py         # Scoring pipeline (AR + MLM tracks)
├── train_ar_pilot.py              # Pilot AR model training
├── test_me_context_diagnostic.py  # Context-dependence diagnostic (pilot)
├── test_me_nonce.py               # Nonce ME evaluation scripts
├── test_me_multiseed.py           # Multi-seed nonce evaluation
├── test_me_nn_init.py             # Nonce embedding init strategies
├── test_nonce_tokenize.py         # BPE tokenisation verification
├── decision_gate_criteria.md      # Pilot go/no-go criteria
└── osf_prereg.md                  # OSF pre-registration (field-by-field)
```

## Reproducing the Results

### Requirements

- Python 3.12
- PyTorch 2.4+ with DirectML (for AMD GPUs) or CUDA
- transformers 5.3.0
- scipy 1.17+

```bash
pip install torch torch-directml transformers scipy numpy
```

### Training the 45-model grid

```bash
python train_grid.py
```

This trains all 45 models (3 sizes × 3 durations × 5 seeds) on AO-CHILDES. Wall time: ~31 hours on AMD RX 7900 GRE.

### Evaluation

```bash
python eval_grid.py
```

Runs the ME battery (suppression, nonce diagnostic, dose-response) on all 45 checkpoints. Outputs `results/grid_eval.json`.

### Statistical analysis

```bash
python analyze_grid.py
```

Runs all pre-registered confirmatory tests (H1–H4) and exploratory analyses. Outputs `results/analysis_report.txt`.

### Corpus analysis

```bash
python corpus_analysis_script.py data/aochildes.txt
```

Computes noun repetition rates and co-occurrence statistics. Outputs `corpus_noun_stats.json`.

## Pre-Registration

The confirmatory experiment (H1–H4) was pre-registered on OSF ([https://osf.io/zu7af](https://osf.io/zu7af)) before any grid models were trained. The pilot model (medium_ep10_s0) was trained before pre-registration; all other 44 models were trained after.

### Pre-registered hypotheses

| Code | Prediction | Test | Result |
|------|-----------|------|--------|
| H1 | Anti-ME rate > 50% in all 9 cells | Sign test, α = 0.05 | **Confirmed** (85–100%) |
| H2 | nonce_only ≥ full_context in all 9 cells | Wilcoxon, α = 0.05 | **Confirmed** (all 9) |
| H3 | >50% monotonic dose-response in all 9 cells | Kendall τ, α = 0.05 | **Partially confirmed** (8/9) |
| H4 | No ME-consistent suppression in any cell | Conjunction of H1 | **Confirmed** |

## Citation

```bibtex
@article{cacioli2026repetition,
  title={Repetition Without Exclusivity: Scale Sensitivity of Referential Mechanisms in Child-Scale Language Models},
  author={Cacioli, JP},
  journal={arXiv preprint},
  year={2026}
}
```

## License

Code: MIT. Data: AO-CHILDES is distributed under its original license via CHILDES (MacWhinney, 2000).
