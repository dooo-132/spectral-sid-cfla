# CFLA: Cross-Factorization Likelihood Aggregation — Anonymous Code Release

> **Anonymous submission.** This repository is a code-only artifact accompanying
> a double-blind paper submission. It contains no author, institution, dataset, checkpoint, result table or
experiment artifact; a few frozen reference metric values are embedded in
verification-only code (labelled as such, not as shipped results). Every command below is annotated with what
> it needs: `[runs as-is]`, `[needs your data]`, `[needs retraining]` or
> `[reference only]` (see §10 for the definitions).

---

## 1. Overview

**CFLA (Cross-Factorization Likelihood Aggregation)** is an inference-time
reranking method for generative recommendation. It treats every Semantic-ID
(SID) system as a *factorization of item identity*, i.e. a mapping

```
F : I -> V_F^{L_F},      i -> [c_{i,1}, ..., c_{i,L_F}],
```

and exploits several such factorizations **of the same item universe and the
same item embeddings** instead of committing to a single SID.

Given a heterogeneous pool of factorizations **F = {F_1, ..., F_M}** (the paper's
main pool is `M5 = Spectral-K2, Spectral-K3, Spectral-K4, Spectral-K5, RQ-VAE5`):

1. **Candidate generation.** For every factorization `F` and user `u`, beam
   search (beam size 30 in the paper) generates candidate SIDs; each candidate
   is decoded back to an item id to form `C_F(u)`.
2. **Shared candidate union.** `C(u) = ∪_F C_F(u)`. Candidates are only ever the
   *generated* items; the ground-truth target is never inserted.
3. **Teacher-forced candidate NLL.** For every candidate `i` in the union and
   every factorization `F`, the **complete** SID `SID_F(i)` is teacher-forced
   through the model `θ_F`, conditioned on the history re-encoded under `F`, and
   the per-token negative log-likelihoods are summed over all `L_F` positions:

   ```
   NLL_F(u,i) = -Σ_{l=1..L_F} log P_{θ_F}( c_{i,l} | S_F(u), c_{i,<l} )
   ```

   This step is target-free and can score a candidate under a factorization that
   did not generate it.
4. **Validation-fitted calibration.** Raw sequence NLLs are not comparable
   across factorizations (different lengths, token distributions, model
   calibration):

   ```
   z_F(u,i) = (NLL_F(u,i) - μ_F) / σ_F,
   ```

   where `μ_F`, `σ_F` are estimated on **all validation candidates** and frozen
   for test.
5. **Log-sum-exp aggregation.** `S_CFLA(u,i) = log Σ_F exp(-z_F(u,i))`, sorted
   descending; the final top-R list is returned. The aggregation has no
   trainable parameters.

The paper's method section describes this pipeline; this artifact provides the
implementation (`src/`, `configs/`) and the reproduction entry points
(`scripts/`), but none of the data, embeddings, SID arrays, checkpoints or
results.

**What is in this release**

* Source code: spectral SID construction, SID tokenizers/generators (RQ-VAE5,
  RQ2-OPQ3, even/random dimension split), T5 training and beam dumping,
  teacher-forced candidate-NLL scoring, calibration + LSE aggregation,
  item-level evaluation, the paper's M5 pipeline drivers, and supporting
  analysis/ablation code.
* Configs for the canonical settings.
* Data-format / item-id / SID-format / protocol / environment documentation.
* CPU unit tests and a synthetic end-to-end demo.

**What is NOT in this release:** Amazon Reviews 2014 data, preprocessed splits,
item embeddings, SID arrays, model checkpoints, beam dumps, prediction caches,
the paper source/PDF and any result tables. They are inputs you prepare or
recreate (§4–§8).

---

## 2. Repository Structure

```
.
├── README.md
├── LICENSE                        MIT, copyright holder anonymized (see §11)
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── canonical.yaml             canonical SID/model/training/inference constants
│   ├── canonical_full.yaml
│   ├── experiments/               16 per-dataset/per-K spectral configs (K=2..5 x 4 datasets)
│   └── paper/RQ5_RECIPE_FINAL.json   frozen RQ-VAE5 tokenizer recipe
├── src/
│   ├── sid/                       spectral SID construction: affinity, spectral
│   │                              partition, subspace quantization, collision token,
│   │                              io, CLI generator
│   ├── tiger/                     T5 recommender: dataset, dataloader, train, dump_beams
│   ├── fusion/                    item-level metrics, uniform fusion, canonicalisation
│   ├── analysis/                  teacher-forced scoring core, per-user metrics
│   ├── utils/                     path resolution (env vars), config schema, run metadata
│   ├── runner.py                  unified per-configuration SID→train→beam→eval runner
│   ├── data_prep/                 raw Amazon Reviews 2014 → parquet + item embeddings
│   ├── tokenization/
│   │   ├── paper/                 RQ-VAE5 tokenizer driver, RQ2-OPQ3 generator
│   │   └── legacy_rqvae/          RQ-VAE reference implementation + dimension-split
│   │                              generator + collision-token append
│   ├── training/                  paper-pool 5+1 T5 training driver
│   ├── legacy_audited/            historical fusion utilities kept for reference
│   └── cfla/
│       ├── paper_pipeline/        paper M5 pipeline: candidate union / candidate-NLL
│       │                          scoring / CFLA wrapper / single-model table /
│       │                          efficiency / codebook sensitivity drivers
│       ├── paper_results/         table/figure builders for the paper numbers
│       ├── final_ablation/        supporting ablation code (earlier two-factor study)
│       ├── candidate_conditioned/, cross_tokenizer_shared/,
│       ├── factorization_generality/, multifactor_fusion/,
│       ├── sports_confirmation/, adaptive_factorization/,
│       └── bootstrap/            paired user-level bootstrap export
├── scripts/                       thin shell wrappers for every pipeline stage
├── examples/synthetic_demo.py     CPU-only synthetic demo (SID construction + CFLA math)
├── tests/                         unit tests (synthetic fixtures; no data needed)
└── docs/
    ├── DATA_FORMAT.md             expected input layout; parquet/beam-dump formats
    ├── SID_FORMAT.md              SID array + slot-wise token encoding
    ├── ITEM_ID_CONVENTION.md      the item-id spaces (read before writing code)
    ├── EXPERIMENT_PROTOCOL.md     canonical protocol (metrics, fusion, statistics)
    ├── ENVIRONMENT.md             verified environment + environment variables
    ├── PIPELINE_STAGES.md         stage → script → entry point → runnability map
    ├── SID_REPRODUCTION_REPORT.md validation evidence for the spectral SID code
    ├── CODE_MAPPING.md            paper component ↔ code mapping (file:line)
    ├── RELEASE_VALIDATION.md      what was executed / statically checked / not run
    ├── ANONYMITY_AUDIT.md         double-blind anonymity audit
    └── UPLOAD_CHECKLIST.md        pre-upload checklist and inventory
```

`docs/CODE_MAPPING.md` maps each paper component to the exact files and entry
points; `docs/PIPELINE_STAGES.md` is the practical stage-by-stage map.

---

## 3. Environment Setup

**Verified environment** (import + CPU-only synthetic test runs):

| component | version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.12.0 (verified CUDA 13.0 build: `2.12.0+cu130`) |
| Transformers | 5.9.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| PyYAML | 6.0.2 |
| tqdm | 4.67.3 |
| matplotlib | 3.9.2 |
| faiss-cpu | 1.14.2 (optional: RQ-OPQ generator) |
| sentence-transformers | 5.5.1 (optional: item-embedding preparation) |

```bash
python3 -V                       # 3.12.x
python3 -m pip install -r requirements.txt

# torch wheels are driver/CUDA specific: requirements.txt pins 2.12.0, which
# resolves to the CPU wheel. For GPU training install the CUDA build matching
# your driver from https://pytorch.org (the verified build was 2.12.0+cu130).
```

**Hardware.** The paper's training and latency measurements used an NVIDIA
RTX 4090 (24 GB class) with CUDA 13.0. T5 training and beam generation need a
GPU; unit tests, the synthetic demo, SID construction on small inputs and all
metric math run on CPU.

**Path configuration.** Every path comes from an environment variable; no
absolute path is hard-coded and all defaults are relative to the repository:

```bash
export CFLA_ARTIFACT_ROOT=/path/to/artifacts   # inputs you prepare (read-only), default <repo>/artifacts
export CFLA_OUTPUT_ROOT=/path/to/outputs      # all new artifacts,            default <repo>/outputs
# optional fallback root consulted when a file is missing from CFLA_ARTIFACT_ROOT:
export CFLA_UPSTREAM_ROOT="${CFLA_ARTIFACT_ROOT}"
```

The expected input layout is documented in `docs/DATA_FORMAT.md`.

**Sanity checks** `[runs as-is]` (CPU, no data):

```bash
bash scripts/check_environment.sh     # versions + resolved path roots
bash scripts/smoke_test.sh            # compile + unit tests + synthetic demo
python3 examples/synthetic_demo.py    # standalone synthetic demo
```

---

## 4. Dataset Preparation

**No dataset, embedding, SID array or checkpoint is shipped.**

The experiments use the **Amazon Reviews 2014** benchmark, distributed by the
original authors:

* Dataset page: <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>
* Third-party data with its own terms of use; obtain it from the distributor
  and cite the dataset. Do not redistribute it with this code.

Categories used in the paper: `Beauty`, `Cell_Phones_and_Accessories`,
`Health_and_Personal_Care`, `Toys_and_Games` (development) and
`Sports_and_Outdoors` (confirmatory study).

### 4.1 Raw files

Per category, download the 5-core review file and the metadata file, e.g. for
Beauty: `reviews_Beauty_5.json.gz`, `meta_Beauty.json.gz` (substitute the
category name for the others). The review file is one JSON object per line with
at least `reviewerID`, `asin`, `unixReviewTime`; the metadata file has one JSON
object per line with `asin`, `title`, `brand`, `categories`, `price`.

### 4.2 Preprocessing `[needs your data]`

```bash
export CFLA_RAW_DATA=/path/to/raw          # holds reviews_*/meta_* files
bash scripts/prepare_data.sh --dataset Beauty --stage interactions
bash scripts/prepare_data.sh --dataset Beauty --stage embeddings \
    --model-dir /path/to/sentence-t5-base
```

The protocol implemented in `src/data_prep/` is:

1. **Parsing.** Read `reviews_<D>_5.json.gz` line by line; keep
   `(reviewerID, asin, unixReviewTime)`.
2. **Integer mappings (1-based).** Map reviewer ids to `1..N_users` and ASINs to
   `1..N_items`; save `user_mapping.npy`, `item_mapping.npy`.
3. **Chronological histories.** Group interactions by user, sort by
   `unixReviewTime`.
4. **Leave-one-out splits** (same users, consecutive chronological segments):
   `train = history[:-2]`, `valid = history[:-1]`, `test = history`; each split
   row is `history` + next item as `target`.
5. **Parquet splits.** `train.parquet`, `valid.parquet`, `test.parquet` with
   columns `user` (int, 1-based), `history` (list[int], 1-based, chronological),
   `target` (int, 1-based).
6. **Item embeddings.** Build one text per item from
   `title / brand / categories / price`, encode it with
   `sentence-transformers/sentence-t5-base` (768-d), save `item_emb.parquet`
   with columns `itemID` (1-based) and `embedding`.

For a CPU-only wiring check, `--synthetic-smoke` generates a tiny **synthetic**
raw file and **synthetic** embeddings (labelled as such) instead of using the
real dataset.

Expected layout after preparation (see `docs/DATA_FORMAT.md`):

```
${CFLA_ARTIFACT_ROOT}/data/<Dataset>/
├── train.parquet  valid.parquet  test.parquet
├── item_mapping.npy  user_mapping.npy
├── item_emb.parquet                                     # SID-construction input
└── <Dataset>_t5_spectral_<affinity>_k<K>_sid<L>.npy     # SID arrays (built in §5)
```

> **Item-id convention (critical, see `docs/ITEM_ID_CONVENTION.md`).** Dataset
> item ids are **1-based** (`1..N`, parquet and `item2code` keys); SID array row
> index is **0-based** (`row = item - 1`); historical beam dumps store 0-based
> ids with `-1` = invalid. Canonicalise raw-dump ids with
> `src.fusion.common.canonicalize_item_ids` before using them.

---

## 5. Semantic ID Construction

### 5.1 Spectral SIDs (the paper's `Spectral-K` factorizations) `[needs your data]`

Given the item embedding matrix `X ∈ R^{N×d}`, `src/sid/` performs:

1. **Pearson affinity between embedding dimensions.** For each dimension pair
   `(p,q)`, compute the Pearson correlation `ρ_pq` over all items and set
   `A_pq = (ρ_pq + 1)/2`, `A_pp = 1` (non-negative affinity).
2. **Normalized Laplacian** `L = I - D^{-1/2} A D^{-1/2}`, `D_pp = Σ_q A_pq`.
3. **Spectral partition.** Take the eigenvectors of the `K` smallest
   eigenvalues and run k-means on the rows of that spectral embedding → the `d`
   dimensions are partitioned into `K` subspaces.
4. **Subspace quantization.** In each subspace, whiten (PCA, `pca_dim = 64` by
   default), run k-means with codebook `C = 256`, and use the nearest-centroid
   index as that subspace's semantic token.
5. **Disambiguation token.** Items sharing the same base code receive a
   deterministic per-collision-group local index, appended as the last token so
   the full item→SID map is injective. SID length is `L = K + 1`.

Canonical parameters (do not change for reproduction): affinity
`nonneg_pearson` (`(ρ+1)/2`), `K ∈ {2,3,4,5}`, `codebook_size = 256`,
`pca_dim = 64`, seed `2025`, `SpectralClustering(affinity="precomputed",
assign_labels="kmeans", n_init=20)`, `KMeans(n_init=30, max_iter=500)`.
`docs/SID_REPRODUCTION_REPORT.md` documents the verification that this
implementation reproduced the paper's frozen SID arrays bit-for-bit on Beauty.

```bash
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 2
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 5
# or: python3 -m src.sid.generator --dataset Beauty --affinity nonneg_pearson \
#         --k 2 --codebook_size 256 --seed 2025 --pca_dim 64
```

Outputs go to `${CFLA_OUTPUT_ROOT}/sids/<Dataset>/<affinity>/k<K>/seed<seed>/`
(`sid.npy`, `base_sid.npy`, `dim_groups.npy`, `dim_labels.npy`,
`metadata.json`, `manifest.sha256`). The writer refuses to write inside
`CFLA_ARTIFACT_ROOT`.

**SID array / token format** (see `docs/SID_FORMAT.md`): `.npy`, `int64`, shape
`(n_items, L)`, codes in `[0,255]`; SID row `i` corresponds to dataset item
`i+1`. Tokens use a slot-wise flat vocabulary `token = code + position*256 + 1`,
token 0 is padding/EOS, so `vocab_size = 256*L + 1` (769/1025/1281/1537 for
`L = 3/4/5/6`).

### 5.2 RQ-VAE5 (paper pool member) `[needs retraining]`

RQ-VAE with **5 residual-quantization levels** (codebooks `[256]*5`) followed by
one disambiguation token (`L = 6`). The frozen recipe is
`configs/paper/RQ5_RECIPE_FINAL.json`. The driver
`src/tokenization/paper/train_tokenizer_5plus1.py` trains the tokenizer and
writes the SID array; `src/tokenization/legacy_rqvae/` contains the RQ-VAE
reference implementation it invokes.

```bash
bash scripts/build_sid.sh --tokenizer rqvae5 --dataset Beauty
```

### 5.3 RQ-OPQ / RQ2-OPQ3 (single-model comparison row) `[needs your data]`

The paper's `RQ-OPQ` row is **RQ2-OPQ3**: 2 residual-quantization positions
followed by 3 OPQ/product-quantization positions (5 semantic + 1 disambiguation
token, `L = 6`), following the OneSearch-style RQ + OPQ construction. The
generator is `src/tokenization/paper/generate_rq2_opq3_sid.py` and needs `faiss`
(`faiss-cpu` in `requirements.txt`).

```bash
bash scripts/build_sid.sh --tokenizer rq2_opq3 --dataset Beauty
```

### 5.4 Even Split and Random Split (single-model comparison rows) `[needs your data]`

Both partition the `d` embedding dimensions into `K` groups and then quantize
each subspace exactly as in §5.1 (subspace PCA whitening + k-means, codebook
256, plus disambiguation token):

* **Even Split** — contiguous equal-width dimension blocks.
* **Random Split** — a seeded random permutation of dimensions split into `K`
  groups.

```bash
bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty --method even
bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty --method random
```

> **Note.** The paper's single-model comparison is length-matched at 5 semantic
> + 1 disambiguation token: `RQ-VAE5`, `RQ2-OPQ3`, `Even-Split-K5`,
> `Random-Split-K5`, `Spectral-K5` are all `L = 6`. The CFLA pool is
> heterogeneous (`L = 3..6`), so no length matching is claimed there.

---

## 6. Training

One T5 encoder–decoder is trained **per factorization**. Architecture (fixed):
4 encoder + 4 decoder layers, `d_model = 128`, `d_ff = 1024`, 6 heads,
`d_kv = 64`, dropout 0.1, T5 `relu` feed-forward, `pad = eos = 0`. Training:
batch 256, lr `1e-4`, up to 200 epochs, early-stopping patience 10, maximum
history length 20, seed **2025**; checkpoint selection uses **item-level
validation NDCG@20** (`--eval_mode item`, the default and the paper's metric).

Inputs required (not shipped): the parquet splits of §4 and the SID `.npy` array
for that factorization from §5. The vocabulary must satisfy
`vocab_size = 256*L + 1`.

```bash
# spectral single model (SID -> train -> validated checkpoint)
bash scripts/train.sh --model t5_single --dataset Beauty --k 2 \
    --code-path "${CFLA_OUTPUT_ROOT}/sids/Beauty/nonneg_pearson/k2/seed2025/sid.npy"

# paper-pool 5+1 tokenizer model (RQ-VAE5 / RQ1-OPQ4 / OPQ5 arms)
bash scripts/train.sh --model t5_5plus1 --dataset Beauty --arm rqvae5
```

`--code-path` is the SID array produced by `scripts/build_sid.sh` (under
`${CFLA_OUTPUT_ROOT}/sids/<Dataset>/<affinity>/k<K>/seed<seed>/sid.npy`); if it is
omitted, the wrapper falls back to the canonical artifact-root path.

The low-level entry points behind the wrappers are
`python3 -m src.tiger.train ...` (T5 training) and
`python3 src/training/train_t5_5plus1.py ...` (paper-pool driver); run either
with `--help` for the full flag list. Everything is written under
`${CFLA_OUTPUT_ROOT}/runs/` (or the driver's own output directory); nothing is
written into `CFLA_ARTIFACT_ROOT`.

The full paper-scale training set is: one T5 per factorization — 5 spectral
factorizations (`K = 2,3,4,5`) + RQ-VAE5 for the M5 pool, plus the length-matched
single-model baselines — on four development datasets, plus the RQ-VAE5
tokenizer's own training run. `[needs retraining]`

---

## 7. Inference and CFLA

### 7.1 Candidate generation per factorization `[needs retraining]`

For each trained model, beam search generates `beam_size` candidate SID
sequences per user, decodes each SID to an item id, and dumps the beams:

```bash
bash scripts/generate_candidates.sh --dataset Beauty --k 2 --split test
```

(real entry point: `python3 -m src.tiger.dump_beams`; repeat for every
factorization and for `--split valid` and `--split test`). The dump format is
documented in `docs/DATA_FORMAT.md`: `candidate_item_ids` `(n_users, beam)`
0-based (`-1` = invalid), `candidate_tokens` `(n_users, beam, L)`,
`sequence_scores` (**length-normalized log-probabilities**, `Σ logp / L`;
softmax before using them as probabilities), `transition_logprobs`,
`target_item_ids` and metadata.

### 7.2 Shared candidate union — ground truth never inserted

The union `C(u) = ∪_F C_F(u)` is built from the generated candidates of all
pool factorizations only. No stage reads the target when constructing the pool;
the target is used only afterwards by the evaluation metrics. Code:
`src/cfla/paper_pipeline/build_m5_rq5.py` (candidate/beam pooling for the M5
pool) and `src/cfla/paper_pipeline/build_m5_cfla.py` (CFLA score computation
over the same union). The generic union builder is
`src/cfla/multifactor_fusion/mf_common.py` (`build_union`).

### 7.3 Teacher-forced complete-SID NLL per factorization `[reference only]`

For every candidate in the union and every factorization, the complete SID is
teacher-forced under the history encoded in that factorization; the score is the
sum of the `L_F` per-position negative log-likelihoods (`TotalNLL_F(u,i)`) — see
`src/analysis/teacher_forced.py` for the scoring core and
`src/cfla/paper_pipeline/score_m5_candidate_nll.py` for the paper-pipeline
driver (`--datasets --splits valid,test --factorizations ... --device cuda`,
plus `--check`, `--verify-sample`, `--require-convention-match` guards).

### 7.4 Validation-fitted zNLL calibration and LSE aggregation `[reference only]`

For each factorization `F`, fit `μ_F, σ_F` on **all validation candidates** and
freeze them; then `z_F(u,i) = (NLL_F(u,i) − μ_F)/σ_F` and
`S_CFLA(u,i) = logsumexp_F(−z_F(u,i))`. Ranking is by descending `S_CFLA`; the
aggregation has no trainable parameters. The calibration statistics come only
from validation, never from test.

The released reference implementation of the CFLA scoring primitives is
`src/cfla/multifactor_fusion/mf_core.py` (`fit_scalers`, `z_of`,
`score_lse_from_znll`, `rank_and_metrics`); the paper-pipeline drivers
(`build_m5_rq5.py` / `score_m5_candidate_nll.py` / `build_m5_cfla.py`) contain
their own frozen equivalents. A self-contained demonstration of the math on
synthetic NLLs runs in `examples/synthetic_demo.py` `[runs as-is]`.

### 7.5 Full inference flow

```bash
# after training + beam dumps: candidate union -> candidate NLL -> calibration
# -> CFLA aggregation -> evaluation, via the paper-pipeline drivers
bash scripts/run_cfla.sh --stage candidate_nll --datasets Beauty --device cuda
bash scripts/run_cfla.sh --stage m5            --datasets Beauty
bash scripts/run_cfla.sh --stage m5_cfla       --datasets Beauty
```

These stages consume the authors' frozen artifacts (beam dumps, SID arrays,
checkpoints, candidate-NLL parquet) and are `[reference only]`: they document
exactly how the paper's numbers were produced but cannot be executed from a
fresh clone without those artifacts (see §10 and `docs/PIPELINE_STAGES.md`).

---

## 8. Evaluation

The paper reports **item-level** Recall@K and NDCG@K for `K ∈ {5, 10, 20}`:

* decode candidate tokens to item ids (SID→item inverse map);
* canonicalise raw beam-dump ids (0-based → 1-based, `-1` sentinel preserved);
* merge beams that decode to the same item by summing their beam probabilities;
* rank items by the fused score (Uniform or CFLA), compute per-user Recall@K /
  NDCG@K and average over test users;
* significance: paired **user-level** bootstrap with 5,000 resamples (95% CI,
  two-sided p-value), paired on identical users.

`--eval_mode item` is the default and the correct metric; the legacy token-prefix
proxy (`--eval_mode token`) is not used for reported numbers.

```bash
# evaluate one beam dump directly [needs your data]
bash scripts/evaluate.sh --dump "${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/test_k5_sid6.npz" \
    --out "${CFLA_OUTPUT_ROOT}/metrics/eval_Beauty_k5_test.json"
# real entry point used by the wrapper: python3 scripts/evaluate.py --dump ...
```

The reference-only table builders for the paper numbers are
`src/cfla/paper_results/{build_table1_final,make_table_efficiency,make_fig_codebook}.py`.

---

## 9. Ablation Studies

The current manuscript reports **no separate ablation section**; the studies
below are the analyses it does report (main comparison, robustness, efficiency)
plus supporting ablation code from an **earlier two-factor study** that is
shipped separately and clearly labelled. Nothing here can be executed without
the inputs named per item.

### 9.1 Studies reported in the paper

**(a) Single-SID vs multi-SID** (Table 1 single-model columns vs CF-Uniform /
CF-LA). The released pipeline builds the length-matched single models (§5.2–5.4,
§6) and the M5 pool (§5.1 + §5.2), generates candidates (§7.1) and evaluates
(§8). `[needs your data + retraining]`; the fused M5 numbers additionally need
the authors' frozen artifacts or a full retraining.

**(b) Cross-factorization rescoring: Uniform vs CFLA** (Table 1 and the
efficiency table). Both methods operate on the *same* candidate union; only the
scorer differs (uniform probability fusion vs teacher-forced zNLL + LSE).
Command chain: `scripts/run_cfla.sh --stage m5` → `--stage m5_cfla`.
`[reference only]` (needs the frozen artifacts or full retraining).

**(c) Codebook sensitivity** (Beauty, `C ∈ {32,64,128,256,512}`, four spectral
factorizations): `src/cfla/paper_pipeline/build_codebook_sensitivity.py`
(`--stage cpu` recomputes from frozen artifacts, `--stage cfla-gpu` needs
checkpoints/GPU, `--self-test` runs a standalone self-check).
`[reference only; --self-test may run standalone]`.

**(d) Efficiency / latency** (96 users, beam 30, CUDA sync, 5 warm-up + 10
measured runs, model loading excluded):
`src/cfla/paper_pipeline/measure_inference_complexity.py` (plus
`src/cfla/paper_results/make_table_efficiency.py`). `[reference only: needs GPU
+ trained checkpoints]`.

### 9.2 Supporting ablation code (earlier two-factor study — not Table 1)

`src/cfla/final_ablation/` and the related `src/cfla/` subdirectories contain the
supporting study behind the paper's mechanism analysis. That study used **two**
factorizations (`F1 = Spectral-K3`, `F2 = RQ-VAE-3layer`, the earlier 3-level
tokenizer, `L = 4`) — an earlier pipeline generation, not the RQ-VAE5 M5
pipeline behind Table 1. It implements:

* **NLL calibration**: z-calibrated LSE vs raw / mean-token / mean-centred LSE
  (`fa_cpu_ablations.py calibration`);
* **Aggregation function**: LSE vs MinNLL / AvgNLL / Uniform (+ a WorstNLL
  negative control) (`fa_cpu_ablations.py aggregation`);
* **Retrieval vs likelihood** hybrid (`fa_cpu_ablations.py retrieval`);
* **Candidate / beam budget**, including a count-matched control
  (`fa_cpu_ablations.py beam`);
* **Subgroup / rescue-damage / bootstrap diagnostics, temperature**
  (`fa_cpu_ablations.py subgroup|rescue|bootstrap|temperature`);
* **Shared vs independent parameters** (`fa_ablation_d.py`, `fa_independent.py`);
* **Same-family vs cross-family pairs** (`fa_family.py`, inventory/analysis);
* **Sports_and_Outdoors confirmation** (`src/cfla/sports_confirmation/`) — this
  study was removed from the current manuscript, so it is supporting evidence
  only.

```bash
bash scripts/run_ablations.sh --list
bash scripts/run_ablations.sh --study final-ablation --cmd calibration
```

`[reference only]`: these scripts recompute the frozen reports from the authors'
frozen beam dumps / SID arrays / checkpoints (`CFLA_ARTIFACT_ROOT`) and in places
from the earlier RQ-VAE3 artifacts. They are not a from-scratch reproduction
path and cannot be executed by a reviewer without those artifacts.

> **Rejected alternatives.** `src/cfla/multifactor_fusion/` also contains
> learned-gate / dynamic-fusion variants that the paper does **not** adopt as the
> method. They are shipped only as supporting/rejected code and must not be
> presented as CFLA.

### 9.3 Not reproducible from code alone

* Efficiency/latency numbers need a GPU and trained checkpoints.
* Mechanism figures/analyses need the authors' frozen analysis artifacts.
* The Sports confirmatory study needs its own data and trained models.
* No results files are shipped, so no number can be regenerated without
  retraining or the authors' artifacts.

---

## 10. Reproducibility Notes

**Runnability labels used above**

| label | meaning |
|---|---|
| `[runs as-is]` | CPU-only, synthetic inputs, no data or checkpoints needed |
| `[needs your data]` | needs your own preprocessed data / SID arrays (`CFLA_ARTIFACT_ROOT`) |
| `[needs retraining]` | needs tokenizer/T5 checkpoints; train first (§6) |
| `[reference only]` | shipped for completeness; consumes the authors' frozen experiment artifacts, which are not distributed |

* **Seeds.** Seed **2025** is used for SID construction (`random_state` of the
  spectral clustering and k-means), T5 training and beam generation; the paired
  bootstrap uses seed `20260912` (see `configs/canonical.yaml`).
* **Determinism.** The spectral partition and k-means are deterministic given
  the seed and canonical `n_init` values (`SpectralClustering n_init=20`,
  `KMeans n_init=30`, `max_iter=500`); changing them is a configuration change,
  not a refactor. GPU T5 training is not bit-deterministic across
  hardware/driver stacks; re-training may reproduce metrics only approximately.
* **Validation-only selection.** Checkpoints are selected by item-level
  validation NDCG@20; `μ_F/σ_F` are fitted on all validation candidates; no
  hyper-parameter is selected on test.
* **What needs retraining vs. not.** SID construction needs the item embeddings
  only (no T5 training). Candidate generation needs trained per-factorization T5
  models. Teacher-forced NLL + calibration + LSE aggregation + evaluation are
  inference-only. The paper-scale full pipeline requires retraining every
  factorization's T5 model on four datasets; the RQ-VAE5 tokenizer also needs
  its own training run.
* **Runtime factors.** T5 training: up to 200 epochs with patience-10 early
  stopping, batch 256; candidate generation: beam 30 per factorization; CFLA's
  additional inference cost is teacher-forcing every union candidate under every
  factorization (the paper's efficiency table quantifies it; not reproducible
  here without a GPU and checkpoints).
* **Version drift.** `requirements.txt` pins the verified versions (§3). A
  different CUDA/driver stack may need a different PyTorch wheel; nothing in the
  code depends on the OS or on absolute paths.

---

## 11. License and Acknowledgments

This release is distributed under the **MIT License** (see `LICENSE`). For
double-blind review the copyright holder is written as **"Anonymous Authors"**;
the real copyright holder and the upstream notices must be restored before a
camera-ready public release.

Third-party components (see `THIRD_PARTY_NOTICES.md`):

* **TIGER — Recommender Systems with Generative Retrieval** (Rajput et al.,
  RecSys 2023): the T5 encoder–decoder training/inference code and semantic-ID
  data plumbing in `src/tiger/`, `src/analysis/teacher_forced.py` and parts of
  `src/fusion/` are derived from the TIGER method.
* **Unofficial PyTorch implementation of TIGER** (MIT): parts of `src/tiger/`,
  `src/sid/` and the RQ-VAE reference implementation in
  `src/tokenization/legacy_rqvae/`; the upstream copyright notice must be
  restored before de-anonymization.
* **OneSearch** (RQ-OPQ reference): the RQ-OPQ generator follows OneSearch's
  public code structure (FAISS k-means residual quantization + OPQ/PQ index
  factory).
* **Amazon Reviews 2014** (J. McAuley and J. Leskovec,
  <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>): the
  experimental datasets. Not included; obtain them from the original
  distributor and comply with its terms.
* **PyTorch / HuggingFace Transformers / NumPy / SciPy / scikit-learn / pandas /
  PyYAML / tqdm / matplotlib / FAISS / sentence-transformers**: distributed
  under their own licenses; verified versions are in `requirements.txt`.

No author, institution or identifying information is included in this artifact
by design.
