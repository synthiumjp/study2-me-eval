# Study 2: Decision-Gate Criteria for ME Pilot
## Date: 2026-03-06
## Status: READY FOR PILOT

---

## Purpose

These criteria define what the pilot must show BEFORE we commit to full training runs
and pre-registration. The pilot runs the scoring pipeline on 1 AR model + 1 BabyBERTa
checkpoint. The goal is NOT to test ME — it's to test whether the evaluation battery
can discriminate ME from null.

---

## Gate 1: Scoring Pipeline Produces Usable Output

### Pass criteria
- [ ] AR model produces finite log-probs for all nonce tokens (no -inf, no NaN)
- [ ] AR model produces finite log-probs for all pseudo-nonce tokens
- [ ] BabyBERTa produces finite log-probs for all familiar nouns at [MASK]
- [ ] BabyBERTa produces finite log-probs for all pseudo-nonce words at [MASK]
- [ ] Log-prob differences between targets are not all identical (i.e., there is variance)

### Fail → action
- If nonce tokens produce -inf: tokenizer.add_tokens() failed; debug embedding initialization
- If all diffs identical: model is ignoring the prompt context; items may be too far OOD
- If BabyBERTa pseudo-nonces return -inf: token not in RoBERTa vocab; replace with alternatives

---

## Gate 2: Baselines Behave As Expected

### Familiar-familiar baseline
- **Expected:** No systematic preference for noun1 vs noun2 across items
- **Pass:** Mean |log P(noun1) - log P(noun2)| < 2.0 nats across 5 items, AND no consistent
  directional bias (i.e., noun1-preferred and noun2-preferred items are roughly balanced)
- **Acceptable:** Mild recency bias (last-mentioned noun preferred by 0.5-1.0 nats) — 
  this is expected and is what the recency control was designed to detect

### Nonce-nonce baseline (AR track only)
- **Expected:** ~chance (no preference for either nonce token)
- **Pass:** Mean |log P(nonce1) - log P(nonce2)| < 1.0 nats across 5 items
- **Concern:** If > 2.0 nats, some nonce tokens may have accidentally meaningful subword 
  structure even after being added whole to the vocab

### Pseudo-pseudo baseline (both tracks)
- **Expected:** ~chance, but with more variance than nonce-nonce (pseudo-nonces have 
  some residual training frequency)
- **Pass:** Mean |diff| < 3.0 nats; no single pseudo-nonce dominates all comparisons
- **Fail → action:** If cork (freq=13) consistently dominates, drop it from pseudo-nonce 
  set and replace with a freq-0 alternative

### Fail → action for baselines
- If baselines show large systematic biases: the item templates are not controlling for 
  surface-level preferences. Need to redesign templates (e.g., balance frequency of frame 
  words, check for n-gram effects from training data).

---

## Gate 3: ME Items Show Discriminable Variance

This is the core gate. We need the ME items to NOT all be at floor or ceiling.

### AR nonce track (Family A, 8 items + 8 swaps)
- **Minimum viable signal:** At least 3/8 items show P(nonce_referent) > P(familiar) 
  (i.e., ME-consistent direction)
- **Floor concern:** If 0/8 or 1/8 ME-consistent — nonce tokens may be too low-probability 
  to ever compete with familiar nouns. Check absolute P(nonce) at completion position.
  - If P(nonce) < 1e-8 for all items: the random embedding init is too far from the 
    learned manifold. Options: (a) init from nearest-neighbor embeddings, (b) do a few 
    gradient steps on nonce tokens with a small set of naming sentences, (c) abandon 
    true nonce track and rely only on pseudo-nonce.
- **Ceiling concern:** If 8/8 ME-consistent with large margins — suspicious. Check that 
  the model isn't just avoiding familiar words in this syntactic frame (i.e., "the X is the" 
  never predicts a previously-mentioned noun regardless of ME context).

### AR pseudo-nonce track (Family A, 6 items + 6 swaps)
- **Minimum viable signal:** At least 2/6 ME-consistent
- **Critical comparison:** If AR nonce track shows ME but AR pseudo-nonce doesn't (or vice 
  versa), the token-type is doing the work, not the disambiguation context. This would 
  require redesign.

### BabyBERTa suppression track (8 items + 8 swaps)
- **Key metric:** fam2_boost (P(cup|ME) - P(cup|baseline))
- **Minimum viable signal:** Mean fam2_boost > 0 across items, with at least 4/8 items 
  showing positive boost.
- **Floor concern:** If all boosts ≈ 0 — the extra labelling sentence ("that is a ball") 
  doesn't affect [MASK] prediction. This could mean:
  - BabyBERTa's context window is too short / attention doesn't carry
  - The syntactic frame is too OOD for BabyBERTa
  - ME is genuinely absent (but can't conclude this until ruling out the above)
- **Ceiling concern:** If fam1_suppression is very large (> 5 nats) but fam2_boost is 
  negative — the model is just suppressing the repeated word, not doing ME-like inference. 
  This is a repetition-avoidance artefact, not ME.

### BabyBERTa pseudo-nonce track (6 items + 6 swaps)
- **Same criteria as AR pseudo-nonce but at [MASK]**

### Fail → action for ME items
- If ALL tracks at floor: item templates are too OOD. Redesign with higher-frequency frames.
- If only nonce track fails: embedding init problem (addressable).
- If only suppression track fails: BabyBERTa may not carry ME — report as null finding for 
  that architecture, continue with AR.
- If variance exists but no ME direction: genuine null finding — proceed to pre-register 
  and run full experiment.

---

## Gate 4: Recency Control Discriminates Real ME from Heuristic

### Diagnostic
For each ME item that shows ME-consistent direction:
- Check its order-swapped twin.
- If the swap ALSO shows ME-consistent direction → genuine ME
- If the swap FLIPS → recency heuristic

### Pass criteria
- **Genuine ME:** ≥ 50% of ME-consistent items survive the recency swap
- **If < 50% survive:** The "ME signal" is actually recency. Need to redesign:
  - Add a delay/distractor between introduction and query
  - Use templates where the familiar noun is mentioned AFTER the nonce
  - Consider three-referent items where recency and ME make different predictions

---

## Gate 5: Cross-Family Convergence (Secondary, Not Blocking)

If Family A passes Gates 3-4, check whether Families B, C, D show consistent patterns.

- **Ideal:** All families show ME in same direction
- **Acceptable:** Family A + at least one other family converge; remaining families show 
  no signal (could be template-specific difficulty)
- **Concern:** If families disagree (A shows ME, C shows anti-ME) — investigate whether 
  specific syntactic frames have unexpected biases

This gate is informational, not blocking. Family A is the primary test.

---

## Gate 6: Synonym Density Effect Direction (BabyBERTa only, Exploratory)

### Expected
ME composite score for 1-label kinds > 2-label kinds (bunny/rabbit, dog/puppy, etc.)

### Interpretation
- **Effect in predicted direction:** Consistent with Byers-Heinlein; the model has learned 
  that some kinds accept multiple labels, weakening ME assumption.
- **No effect:** Model doesn't modulate ME by synonym experience. Still interesting — means 
  ME is either all-or-nothing, or this manipulation is too subtle.
- **Reverse effect:** Unexpected. Would need to check whether synonym pairs have unusual 
  frequency interactions.

This gate is PURELY EXPLORATORY. Does not block proceeding.

---

## Summary Decision Matrix

| Outcome | Decision |
|---------|----------|
| Gates 1-2 pass, Gate 3 shows variance, Gate 4 passes | **GO** — proceed to pre-registration |
| Gates 1-2 pass, Gate 3 at floor for nonce only | **FIX** — address embedding init, re-pilot |
| Gates 1-2 pass, Gate 3 at floor for ALL tracks | **REDESIGN** — items too OOD, need new templates |
| Gates 1-2 pass, Gate 3 variance exists but no ME direction | **GO** — pre-register null-permitting hypothesis |
| Gates 1-4 pass but Gate 4 shows >50% recency | **REDESIGN** — add distractor/delay to templates |
| Gate 1 fails (technical) | **DEBUG** — fix pipeline before any interpretation |
| Gate 2 fails (baselines biased) | **REDESIGN** — templates have uncontrolled confounds |

---

## Pilot Execution Checklist

1. [ ] Set up environment (PyTorch/ROCm, transformers)
2. [ ] Obtain BabyBERTa checkpoint (Huebner et al. HuggingFace or retrain)
3. [ ] Train or load 1 AR model on AO-CHILDES
4. [ ] Run: `python me_scoring_pipeline.py --model_type ar --add_nonce_tokens wug dax blicket toma zib mep fep gorp ...`
5. [ ] Run: `python me_scoring_pipeline.py --model_type mlm ...`
6. [ ] Evaluate against gates 1-6 above
7. [ ] Document results in session log
8. [ ] Make go/redesign/fix decision

---

## Post-Pilot: What Gets Pre-Registered

If GO:
- The exact item battery JSON (frozen)
- The scoring pipeline (frozen, versioned)
- Primary hypotheses:
  - H_ME: AR nonce items show P(nonce_referent) > P(familiar) at above-chance rates
  - H_suppression: BabyBERTa suppression items show P(fam2|ME) > P(fam2|baseline)
  - H_recency: ME signal survives order swap (not reducible to recency)
- Exploratory hypotheses:
  - H_density: 1-label kinds show stronger ME than 2-label kinds
  - H_pseudo: Pseudo-nonce items converge with primary track
  - H_family: ME signal replicates across item families
- Analysis plan: item-level logistic regression (ME-consistent yes/no ~ track + family + seed)
- Decision criteria: pre-registered here, frozen before data collection
