# EXPERIMENT PROTOCOL

## Canonical configuration

Single source of truth for the spectral pipeline: `configs/canonical.yaml`
(SID / model / training / inference constants) and `configs/experiments/*.yaml`
(per-dataset, per-K). Nothing in the code hard-codes an experiment setting: pass a
config or CLI flag and record which one you used. The CFLA pool is described by a
separate pool config (`configs/cfla_pool.example.yaml`).

## Hard rules

1. **Never modify an input artifact.** Inputs are read-only; write new outputs under
   `CFLA_OUTPUT_ROOT`.
2. **Never select hyper-parameters on the test set.** Use validation, or a method
   that requires no selection. CFLA's calibration statistics are fitted on **all
   validation candidates** and frozen before test scoring.
3. **Never insert the ground-truth target into the candidate pool.** Candidate pools
   come from beam generation only; the target is used only for metric computation
   after scoring.
4. **Never compare against a metric computed by a different code path** without
   saying so. The training-time evaluator and the item-level evaluator are different
   estimators; the paper reports the item-level one.

## Model selection metric

`--eval_mode item` (default) selects checkpoints on item-level validation NDCG@20:
decode generated SIDs to item ids, merge beams mapping to the same item by summed
beam probability, rank items.

`--eval_mode token` is the legacy token-prefix proxy; it scores lower and can select
a different checkpoint. It is kept for historical comparison only.

## CFLA protocol

With a pool `F = {F_1, ..., F_M}`:

1. generate `C_F(u)` by beam search for every `F`;
2. `C(u) = ∪_F C_F(u)` (generated candidates only — no target insertion);
3. teacher-force the complete SID of every `i ∈ C(u)` under every `F`; sum the
   per-position negative log-likelihoods → `NLL_F(u,i)`;
4. fit `μ_F, σ_F` on **all validation candidates** and freeze them;
5. `z_F(u,i) = (NLL_F(u,i) − μ_F)/σ_F`;
6. `S_CFLA(u,i) = logsumexp_F(−z_F(u,i))`, rank descending.

Implemented in `src/cfla/cfla.py` and orchestrated by `src/cfla/run.py`
(`scripts/run_cfla.sh`). The runner also computes two baselines on the same union:
`uniform` (mean of per-factorization retrieval probabilities) and `minnll`
(`−min_F z_F`, aggregation-function ablation).

## Evaluation and statistics

Item-level Recall@K / NDCG@K for K ∈ {5,10,20}; targets outside the union contribute
0. Significance: paired **user-level** bootstrap with 5,000 resamples (95% CI,
two-sided p-value, seed 20260912) — `src/cfla/cfla.py::paired_bootstrap`, reported
for CFLA vs Uniform Fusion in `metrics.json`.

## Reproducing a run

```bash
bash scripts/reproduce.sh --stage all --dataset Beauty --raw-dir /path/to/raw
```

This wraps: data preparation → SID construction (spectral + dimension split) →
per-factorization T5 training → beam candidates → CFLA check. RQ-VAE / RQ-OPQ
factorizations require externally built SID arrays (README §5.2–5.3). The full
synthetic pipeline can be validated without any data via
`bash scripts/smoke_pipeline.sh`.
