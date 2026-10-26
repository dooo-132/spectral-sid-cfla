# CFLA: Cross-Factorization Likelihood Aggregation — Anonymous Code Release

> **Anonymous submission.** This repository is a code-only artifact accompanying a
> double-blind paper submission. It contains no author, institution, dataset,
> checkpoint, result table or experiment artifact.

---

## 1. Overview

**CFLA (Cross-Factorization Likelihood Aggregation)** is an inference-time reranking
method for generative recommendation. It treats every Semantic-ID (SID) system as a
*factorization of item identity*,

```
F : I -> V_F^{L_F},      i -> [c_{i,1}, ..., c_{i,L_F}],
```

and uses several such factorizations **of the same item universe** instead of
committing to one SID. Given a pool `F = {F_1, ..., F_M}`:

1. **Candidate generation** — for every `F` and user `u`, beam search generates
   candidate SIDs; each is decoded back to an item id, giving `C_F(u)`.
2. **Candidate union** — `C(u) = ∪_F C_F(u)`. Only generated items are used; the
   ground-truth target is never inserted.
3. **Teacher-forced candidate NLL** — for every `i ∈ C(u)` and every `F`, the
   complete SID `SID_F(i)` is teacher-forced under the history encoded in `F`, and
   the per-token negative log-likelihoods are summed:
   `NLL_F(u,i) = −Σ_l log P_{θ_F}(c_{i,l} | S_F(u), c_{i,<l})`. Target-free.
4. **Calibration** — `z_F(u,i) = (NLL_F(u,i) − μ_F)/σ_F`, with `μ_F, σ_F` fitted on
   **all validation candidates** and frozen for test.
5. **Aggregation** — `S_CFLA(u,i) = log Σ_F exp(−z_F(u,i))`; rank descending. No
   trainable parameters.

**This release is runnable end to end** (dataset downloaded by the user):

```
raw Amazon Reviews 2014
  -> prepare_data.sh   (splits, id maps, item embeddings)
  -> build_sid.sh      (spectral SIDs; even/random dimension-split baseline)
  -> train.sh          (one T5 per factorization)
  -> generate_candidates.sh  (beam dumps for valid/test)
  -> run_cfla.sh       (candidate union -> candidate NLL -> calibration -> LSE -> metrics)
```

`bash scripts/smoke_pipeline.sh` runs this entire chain on a tiny **synthetic**
corpus (CPU only) to verify the code paths without any dataset or GPU.

**Not bundled:** the RQ-VAE and RQ-OPQ tokenizer implementations are third-party /
derived code and are **not shipped**; use the external repositories linked in §5.2–5.3
to build those SID arrays, then pass them to `train.sh --code-path` / add them to the
CFLA pool config. Paper results, datasets, checkpoints and beam dumps are not shipped.

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
│   ├── experiments/               16 per-dataset/per-K spectral configs (K=2..5 x 4 datasets)
│   ├── paper/RQ5_RECIPE_FINAL.json   RQ-VAE5 hyper-parameters (tokenizer not bundled)
│   └── cfla_pool.example.yaml     CFLA factorization-pool config (edit per run)
├── src/
│   ├── sid/                       spectral SID construction (Pearson dimension affinity,
│   │                              spectral partition, subspace k-means, disambiguation
│   │                              token, io, CLI generator)
│   ├── tokenization/dimension_split/  Even/Random dimension-split SID baseline
│   ├── tiger/                     T5 recommender: dataset, dataloader, train, dump_beams
│   ├── analysis/teacher_forced.py teacher-forced scoring core
│   ├── fusion/common.py           item-level metrics, canonicalisation, union helpers
│   ├── cfla/
│   │   ├── cfla.py                CFLA pipeline primitives (union, candidate NLL,
│   │   │                          calibration, LSE, metrics, paired bootstrap)
│   │   └── run.py                 end-to-end CFLA runner (CLI)
│   ├── utils/                     path resolution (env vars), config schema, run metadata
│   └── runner.py                  unified per-configuration SID->train->beam->eval runner
├── scripts/                       stage wrappers (data, SID, train, candidates, CFLA,
│                                  evaluation, reproduction, smoke tests)
└── docs/
    ├── DATA_FORMAT.md             expected input layout; parquet/beam-dump formats
    ├── SID_FORMAT.md              SID array + slot-wise token encoding
    ├── ITEM_ID_CONVENTION.md      the item-id spaces (read before writing code)
    ├── EXPERIMENT_PROTOCOL.md     canonical protocol (metrics, calibration, statistics)
    ├── ENVIRONMENT.md             verified environment + environment variables
    ├── PIPELINE_STAGES.md         stage -> script -> entry point -> inputs
    ├── SID_REPRODUCTION_REPORT.md validation note for the spectral SID implementation
    ├── CODE_MAPPING.md            paper component <-> code mapping
    ├── RELEASE_VALIDATION.md      executed / static-checked / not executed
    ├── ANONYMITY_AUDIT.md         double-blind anonymity audit
    └── UPLOAD_CHECKLIST.md        pre-upload checklist and inventory
```

---

## 3. Environment Setup

**Verified environment** (CPU-only import / synthetic test runs):

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
| sentence-transformers | 5.5.1 (optional: item-embedding preparation) |

```bash
python3 -V                       # 3.12.x
python3 -m pip install -r requirements.txt
# torch wheels are driver/CUDA specific: requirements.txt pins 2.12.0 (CPU wheel);
# for GPU training install the CUDA build matching your driver from https://pytorch.org
```

**Hardware.** T5 training and beam generation need a GPU for paper-scale runs; all
stages also run on CPU for small/synthetic inputs (the smoke pipeline does).

**Path configuration.** No absolute path is hard-coded; everything is resolved from
environment variables with repository-relative defaults:

```bash
export CFLA_ARTIFACT_ROOT=/path/to/inputs    # user-prepared data/SID/checkpoints, default <repo>/artifacts
export CFLA_OUTPUT_ROOT=/path/to/outputs     # all new artifacts,                 default <repo>/outputs
export CFLA_RAW_DATA=/path/to/raw            # raw Amazon Reviews 2014 files,     default <repo>/raw
export CFLA_SENTENCE_T5_DIR=/path/to/sentence-t5-base   # embedding model (optional)
```

---

## 4. Dataset Preparation

**No dataset is shipped.** The experiments use the **Amazon Reviews 2014** benchmark:

* Dataset page: <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>
* Categories: `Beauty`, `Cell_Phones_and_Accessories`, `Health_and_Personal_Care`,
  `Toys_and_Games` (development), `Sports_and_Outdoors` (confirmatory study).
* Third-party data with its own terms; download it yourself and do not redistribute it.

Per category, place the 5-core review file and the metadata file under `CFLA_RAW_DATA`:
`reviews_<Dataset>_5.json.gz`, `meta_<Dataset>.json.gz`.

```bash
bash scripts/prepare_data.sh --dataset Beauty --stage interactions
bash scripts/prepare_data.sh --dataset Beauty --stage embeddings \
    --model-dir /path/to/sentence-t5-base
```

The preprocessing implemented in `src/data_prep/` (upstream TIGER protocol):
1. parse `reviews_<D>_5.json.gz` (one JSON review per line: `reviewerID`, `asin`,
   `unixReviewTime`);
2. map reviewer ids and ASINs to **1-based** integers (`user_mapping.npy`,
   `item_mapping.npy`);
3. build per-user chronological item sequences (sorted by `unixReviewTime`);
4. leave-one-out splits: `train = history[:-2]`, `valid = history[:-1]`,
   `test = history`; each row = `history` + `target`;
5. save `train/valid/test.parquet` with columns `user` (1-based), `history`
   (list of 1-based item ids), `target` (1-based);
6. build one text per item from `title/brand/categories/price`, encode it with
   `sentence-transformers/sentence-t5-base` (768-d), save `item_emb.parquet`
   (`itemID`, `embedding`).

Artifact layout after preparation (see `docs/DATA_FORMAT.md`):

```
${CFLA_ARTIFACT_ROOT}/data/<Dataset>/
├── train.parquet  valid.parquet  test.parquet
├── item_mapping.npy  user_mapping.npy
└── item_emb.parquet                     # SID-construction input
```

> **Item-id convention (critical).** Dataset item ids are **1-based**; SID-array row
> index is **0-based** (`row = item - 1`); beam dumps store **0-based** ids with
> `-1` = invalid. See `docs/ITEM_ID_CONVENTION.md`.

---

## 5. Semantic ID Construction

### 5.1 Spectral SIDs `[runs on CPU / GPU; needs your data]`

Given the item embedding matrix `X ∈ R^{N×d}`, `src/sid/` performs:

1. **Pearson affinity between embedding dimensions** — `A_pq = (ρ_pq + 1)/2`,
   `A_pp = 1`;
2. **normalized Laplacian** `L = I − D^{−1/2} A D^{−1/2}`;
3. **spectral partition** — eigenvectors of the `K` smallest eigenvalues, k-means on
   the dimension rows → `K` dimension groups;
4. **subspace quantization** — per-group PCA whitening (`pca_dim=64`) + k-means
   (`codebook_size=256`); the nearest-centroid index is the semantic token;
5. **disambiguation token** — a deterministic per-collision-group local index is
   appended, making the item→SID map injective. SID length `L = K + 1`.

Canonical parameters: affinity `nonneg_pearson`, `K ∈ {2,3,4,5}`,
`codebook_size=256`, `pca_dim=64`, seed `2025`,
`SpectralClustering(affinity='precomputed', assign_labels='kmeans', n_init=20)`,
`KMeans(n_init=30, max_iter=500)`.

```bash
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 2
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 5
```

Outputs go to `${CFLA_OUTPUT_ROOT}/sids/<Dataset>/<affinity>/k<K>/seed2025/`
(`sid.npy`, `base_sid.npy`, `dim_groups.npy`, `dim_labels.npy`, `metadata.json`,
`manifest.sha256`). SID arrays are `int64`, shape `(n_items, L)`, row `i` = item
`i+1`, codes in `[0,255]`; tokens use the slot-wise layout
`token = code + position*256 + 1`, `vocab_size = 256*L + 1`
(see `docs/SID_FORMAT.md`).

### 5.2 RQ-VAE SIDs `[not bundled — use an external repository]`

The paper's RQ-VAE5 factorization is a 5-level residual-quantization tokenizer
(codebooks `[256]*5`) plus one disambiguation token (`L = 6`). **This release does not
ship an RQ-VAE implementation.** Build the SID array with a public implementation, e.g.

* **RQ-VAE official implementation** — <https://github.com/kakaobrain/rq-vae-transformer>
  (see the repository for its licence and usage).

The hyper-parameters used by the paper are recorded in
`configs/paper/RQ5_RECIPE_FINAL.json` (5 levels, codebook 256, `sk_epsilons =
[0,0,0,0,0.003]`, `vocab_size = 1537`). The SID array you produce must follow the
same format as §5.1: `int64`, shape `(n_items, 6)`, row `i` = item `i+1`, codes in
`[0,255]`, semantic positions first and the disambiguation code last.

### 5.3 RQ-OPQ SIDs `[not bundled — use an external repository]`

The paper's `RQ-OPQ` row is **RQ2-OPQ3**: 2 residual-quantization positions followed
by 3 OPQ/product-quantization positions, plus one disambiguation token (`L = 6`),
following the OneSearch construction (FAISS k-means residual quantization + an
`OPQ,IVF1,PQ` index factory). **This release does not ship an RQ-OPQ implementation.**
Usable public components:

* **FAISS** (OPQ / product quantization) — <https://github.com/facebookresearch/faiss>
* **RQ-VAE official implementation** (residual quantization) —
  <https://github.com/kakaobrain/rq-vae-transformer>
* **OneSearch paper** (construction reference) — <https://arxiv.org/abs/2212.01354>

Produce an `(n_items, 6)` SID array in the same format as §5.1 and use it like any
other factorization.

### 5.4 Even/Random dimension split `[runs as-is; needs your data]`

The paper's single-model comparison includes an even and a random partition of the
embedding dimensions into 5 groups, each quantized like §5.1 plus one disambiguation
token (`L = 6`). The paper-owned implementation is shipped:

```bash
bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty --method even
bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty --method random
```

> **Length matching.** The paper's single-model comparison is length-matched at
> 5 semantic + 1 disambiguation token (`L = 6`): RQ-VAE5, RQ2-OPQ3, Even-Split-K5,
> Random-Split-K5 and Spectral-K5. The CFLA pool is heterogeneous (native lengths
> 3–6), so no length matching is claimed there.

---

## 6. Training

One T5 encoder–decoder is trained **per factorization**. Architecture: 4 encoder +
4 decoder layers, `d_model=128`, `d_ff=1024`, 6 heads, `d_kv=64`, dropout 0.1, T5
`relu` feed-forward, `pad=eos=0`. Training: batch 256, lr `1e-4`, up to 200 epochs,
early-stopping patience 10, maximum history length 20 items, seed `2025`; checkpoint
selection by **item-level validation NDCG@20** (`--eval_mode item`, default).

Inputs: the parquet splits of §4 and one SID array per factorization (§5). The
vocabulary must satisfy `vocab_size = 256*L + 1`.

```bash
# canonical spectral factorization (K=2, L=3)
bash scripts/train.sh --model t5_single --dataset Beauty --k 2

# any other SID system (e.g. an externally built RQ-VAE5 array with L=6)
bash scripts/train.sh --model t5_single --dataset Beauty \
    --code-path /path/to/Beauty_t5_rqvae5.npy --sid-len 6
```

`scripts/train.sh` writes the checkpoint and log under
`${CFLA_OUTPUT_ROOT}/runs/`; the low-level entry point is
`python3 -m src.tiger.train --help` (all flags are documented there, including
`--num_workers 0` for restricted/CPU-only hosts). Repeat for every factorization
you want in the CFLA pool. For the paper configuration that is spectral K=2,3,4,5
plus RQ-VAE5 (the latter from an external tokenizer), per dataset.

---

## 7. Inference and CFLA

### 7.1 Candidate generation per factorization

```bash
bash scripts/generate_candidates.sh --dataset Beauty --k 2 --split valid
bash scripts/generate_candidates.sh --dataset Beauty --k 2 --split test
```

Real entry point `python3 -m src.tiger.dump_beams`: beam search (default beam 30)
generates candidate SID sequences per user, decodes each to an item id and writes a
`.npz` dump under `${CFLA_OUTPUT_ROOT}/beam_dumps/`. Dump keys are documented in
`docs/DATA_FORMAT.md` (`candidate_item_ids` 0-based with `-1` = invalid,
`candidate_tokens`, `sequence_scores` = length-normalised log-probabilities,
`transition_logprobs`, `target_item_ids`). The dumper fails loudly if fewer than 90%
of candidates decode to a valid item (token-layout mismatch guard).

### 7.2 CFLA: candidate union, candidate NLL, calibration, aggregation

Create a pool config listing every factorization's SID array, checkpoint and
valid/test beam dumps (`configs/cfla_pool.example.yaml`), then run:

```bash
bash scripts/run_cfla.sh --config configs/cfla_pool.example.yaml --check   # validate paths
bash scripts/run_cfla.sh --config configs/cfla_pool.example.yaml --device cuda
```

The runner (`src/cfla/run.py`, core in `src/cfla/cfla.py`):

1. loads the beam dumps of all factorizations and checks user alignment;
2. builds the shared candidate union per user (generated items only — the target is
   never inserted; each user's per-factorization retrieval probability is kept);
3. for every (user, candidate) row and every factorization, encodes the history in
   that factorization's token space and teacher-forces the candidate's complete SID
   (`src/analysis/teacher_forced.py`), summing `−log p` over all `L_F` positions;
4. fits `μ_F, σ_F` on **all validation candidates** and freezes them;
5. scores the test candidates with three methods on the same union:
   * `cfla` = `logsumexp_F(−z_F)` (the paper's method),
   * `uniform` = mean of the per-factorization retrieval probabilities (baseline),
   * `minnll` = `−min_F z_F` (aggregation-function ablation);
6. writes `metrics.json` (Recall@K / NDCG@K per method, calibration statistics,
   candidate-pool statistics, paired user-level bootstrap of CFLA vs uniform),
   `per_user.csv` and `run_config.json` into the output directory.

The target is used only for metric computation after scoring; no stage reads it
while building the pool or computing scores.

### 7.3 A complete run

```bash
bash scripts/reproduce.sh --stage all --dataset Beauty \
    --raw-dir /path/to/raw            # data -> sid -> train -> candidates -> CFLA check
```

(You must additionally train the RQ-VAE5 arm from an externally built SID array and
add it to the pool config to reproduce the paper's exact 5-factorization pool.)

---

## 8. Evaluation

The paper reports **item-level** Recall@K and NDCG@K for `K ∈ {5,10,20}`:

* decode candidate tokens to item ids and canonicalise them to 1-based ids;
* merge beams that decode to the same item by summing their beam probabilities;
* rank items by the fused score; compute per-user Recall@K / NDCG@K and average over
  users (targets outside the union contribute 0);
* significance: paired user-level bootstrap with 5,000 resamples (95% CI, two-sided
  p-value) — implemented in `src/cfla/cfla.py::paired_bootstrap` and reported by the
  CFLA runner (`--bootstrap 5000`).

`--eval_mode item` is the default and the correct metric; the legacy token-prefix
proxy (`--eval_mode token`) is not used for reported numbers.

Evaluate one beam dump directly:

```bash
bash scripts/evaluate.sh --dump "${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/test_k2_sid3.npz" \
    --out "${CFLA_OUTPUT_ROOT}/metrics/eval_Beauty_k2_test.json"
```

---

## 9. Ablation Studies

The release supports the paper's core comparison and the aggregation-function
ablation directly on the same candidate union:

| study | how |
|---|---|
| **Single-SID vs multi-SID** (paper Table 1) | train/evaluate one factorization (single) versus run the CFLA pool with several factorizations; the SID constructions for the paper baselines are §5.1–5.4 |
| **Cross-factorization rescoring: Uniform vs CFLA** | `run_cfla.sh` reports `cfla`, `uniform` and `minnll` on the *same* union by default; the paired bootstrap of CFLA vs uniform is included in `metrics.json` |
| **Aggregation function** | the runner implements `cfla` (log-sum-exp), `minnll` and `uniform`; add a pool config per setting |
| **Codebook sensitivity** | rerun `build_sid.sh` with different `--codebook-size` (e.g. 32,64,128,256,512) and retrain/evaluate |

The earlier two-factor study code and the frozen-artifact analysis scripts that
accompanied the development of the paper are intentionally **not** included in this
anonymous release; the shipped pipeline is the code needed to rebuild the method from
scratch. No results files are shipped, so no paper number is reproduced by this
release without training.

---

## 10. Reproducibility Notes

* **Seeds** — SID construction, T5 training and beam generation use seed `2025`; the
  paired bootstrap uses `20260912`. SID clustering/k-means are deterministic given
  the seed and canonical `n_init` values; GPU training is not bit-deterministic
  across hardware/driver stacks.
* **Validation-only selection** — checkpoints are selected by item-level validation
  NDCG@20; `μ_F/σ_F` are fitted on all validation candidates; nothing is selected on
  test.
* **What needs retraining** — SID construction needs only the item embeddings;
  candidate generation needs trained per-factorization T5 models; candidate NLL,
  calibration, aggregation and evaluation are inference-only. A full paper-scale run
  requires training one T5 per factorization per dataset (plus the external RQ-VAE
  tokenizer training).
* **Cost** — CFLA additionally teacher-forces every union candidate under every
  factorization; the paper reports roughly 3× the Uniform-Fusion inference latency on
  the same union (see the paper's efficiency table).
* **Smoke validation** — `bash scripts/smoke_test.sh` checks the environment,
  byte-compiles every file and imports the core packages; `bash scripts/smoke_pipeline.sh`
  runs the whole pipeline on synthetic data (CPU). See `docs/RELEASE_VALIDATION.md`.

---

## 11. License and Acknowledgments

This release is distributed under the **MIT License** (see `LICENSE`). For
double-blind review the copyright holder is written as **"Anonymous Authors"**; the
real copyright holder must be restored before a camera-ready public release.

Third-party components (see `THIRD_PARTY_NOTICES.md`):

* **TIGER — Recommender Systems with Generative Retrieval** (Rajput et al., RecSys
  2023, <https://arxiv.org/abs/2305.05065>): the T5 encoder–decoder architecture,
  semantic-ID data plumbing and the generative-recommendation protocol follow this
  method.
* **RQ-VAE** (residual quantization) — official implementation:
  <https://github.com/kakaobrain/rq-vae-transformer>. Not bundled; used to build the
  RQ-VAE SID arrays with the recipe in `configs/paper/RQ5_RECIPE_FINAL.json`.
* **RQ-OPQ / OneSearch** — the paper's RQ2-OPQ3 construction follows the OneSearch
  paper (<https://arxiv.org/abs/2212.01354>); a usable implementation can be assembled
  from FAISS (<https://github.com/facebookresearch/faiss>, OPQ/PQ) and an RQ
  implementation. Not bundled.
* **Amazon Reviews 2014** (J. McAuley and J. Leskovec,
  <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>): the experimental
  datasets. Not included; obtain them from the original distributor.
* **PyTorch / HuggingFace Transformers / NumPy / SciPy / scikit-learn / pandas /
  PyYAML / tqdm / sentence-transformers**: distributed under their own licenses;
  verified versions are in `requirements.txt`.

No author, institution or identifying information is included in this artifact by
design.
