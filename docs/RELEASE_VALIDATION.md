# RELEASE VALIDATION

Date: 2026-10-26. Release: the `spectral-sid-cfla` directory of this bundled artifact.
This document distinguishes **executed**, **static-checked** and **not executed**
items. No paper-scale training or experiment was run.

## 1. Environment used for validation

| item | value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| CUDA | 13.0 on the authors' machine; **all executed checks below are CPU-only** |
| Transformers / scikit-learn / SciPy / NumPy / pandas / pyarrow | 5.9.0 / 1.9.0 / 1.17.1 / 2.4.6 / 3.0.3 / 24.0.0 |
| PyYAML / tqdm / sentence-transformers | 6.0.2 / 4.67.3 / 5.5.1 (optional) |

`requirements.txt` pins the verified versions (11 pins + 1 optional extra).

## 2. Level 1 — static validation (executed)

| check | command | result |
|---|---|---|
| Python syntax, whole tree | in-memory `compile()` over all `.py` | **38 files, 0 failures** |
| Shell syntax | `bash -n` over all `scripts/*.sh` | **10 files, 0 failures** |
| YAML syntax | `yaml.safe_load` over all `configs/**/*.yaml` | **17 files, 0 failures** |
| JSON syntax | `json.load` over `configs/paper/RQ5_RECIPE_FINAL.json` | OK |
| Import graph | `importlib.import_module` for every non-`__init__` module under `src/` | **27/27 importable** |
| CLI help | `bash scripts/<stage>.sh --help` for every wrapper | 8/8 OK; `check_environment.sh` prints OK |
| README file references | every script/config/docs path in README fenced commands | **11/11 exist** |
| Path independence | scan for POSIX root/home and Windows user-profile prefixes | **0 hits** |
| Config parsing | `src.cfla.run --config configs/cfla_pool.example.yaml --check` | resolves `$CFLA_ARTIFACT_ROOT`/`$CFLA_OUTPUT_ROOT`, reports missing inputs correctly |
| No inherited git history | fresh repository, single commit, `git fsck --lost-found` empty | OK |

## 3. Level 2 — lightweight smoke tests (executed, CPU, synthetic only)

**3.1 `bash scripts/smoke_test.sh` → PASSED.** Checks the environment, byte-compiles
every Python file in memory, and imports the core packages (12/12 imports OK).

**3.2 End-to-end synthetic pipeline: `bash scripts/smoke_pipeline.sh` → PASSED.**
It runs, on a tiny synthetic corpus and CPU only:

1. `prepare_data.sh --synthetic-smoke` → 200 synthetic users, 400 items, splits,
   id mappings, synthetic 768-d item embeddings;
2. `build_sid.sh --tokenizer spectral` for K=2 and K=3 → 400/400 unique, injective
   SIDs, metadata + manifest written;
3. `train.sh --model t5_single --device cpu --num-workers 0 --epochs 3` for both
   factorizations → one T5 checkpoint per factorization;
4. `generate_candidates.sh` (beam 12, CPU) → valid/test beam dumps for both
   factorizations;
5. `run_cfla.sh` → candidate union, teacher-forced candidate NLL per factorization,
   validation-fitted calibration, log-sum-exp aggregation, Recall/NDCG evaluation,
   `metrics.json` / `per_user.csv` / `run_config.json` written; all metrics finite
   and in `[0,1]`, paired-bootstrap block produced.

The smoke corpus and the 3-epoch CPU models are random and tiny: **the resulting
numbers are meaningless as results** (the union coverage obtained by the untrained
models is ~0). The test proves the code path executes end to end and writes its
artifacts. It also exercises the dumper's decode-health guard override
(`--allow-low-decode`), which is required only for such synthetic/untrained runs.

## 4. Level 3 — release reproducibility audit

### 4.1 README commands

Every command in README §3–§8 references a shipped script, config or doc. Classes:

* **runs as-is (CPU, synthetic, no data)** — `check_environment.sh`, `smoke_test.sh`,
  `smoke_pipeline.sh`, `run_cfla.sh --check` (path validation), `build_sid.sh --help`,
  etc.
* **needs your data** — `prepare_data.sh` (Amazon Reviews 2014 + sentence-t5-base),
  `build_sid.sh --tokenizer spectral|dimension_split`, `evaluate.sh`,
  `run_cfla.sh` (after training/candidate generation);
* **needs GPU / retraining** — `train.sh`, `generate_candidates.sh` at paper scale;
* **external** — RQ-VAE / RQ-OPQ SID arrays are built with the linked third-party
  repositories (not bundled).

### 4.2 What was NOT executed and why

| item | why |
|---|---|
| real Amazon Reviews 2014 download / preprocessing | dataset is not shipped; user downloads it |
| paper-scale T5 training (4 spectral factorizations + RQ-VAE5 per dataset) | requires GPU and the dataset |
| beam generation / CFLA scoring on the real corpus | requires trained checkpoints |
| reproduction of the paper's numbers | not claimed: no data, checkpoints or results are shipped |

### 4.3 Content changes applied during release preparation

- Removed (at the authors' request) the research-support directories
  (`paper_results/`, supporting ablation/mechanism analysis code), `tests/`,
  `examples/`, and the previously bundled RQ-VAE / RQ-OPQ implementations; the RQ
  tokenizers are now referenced by public repository links (README §5.2–5.3).
- Added a self-contained, runnable CFLA pipeline (`src/cfla/cfla.py`,
  `src/cfla/run.py`, `scripts/run_cfla.sh`, `configs/cfla_pool.example.yaml`) and the
  synthetic end-to-end smoke pipeline (`scripts/smoke_pipeline.sh`).
- Removed the legacy two-factor study config; updated all docs to the shipped layout.
- Fixed defects found during validation: `dump_beams` package imports; data-prep
  embedding CLI; `num_workers` option for CPU/restricted hosts; checkpoint fallback
  when validation NDCG@20 never exceeds 0 on tiny/synthetic data; CFLA config path
  expansion defaults; stale path references in docs.

## 5. Honesty statement

No paper-scale experiment was run for this release and no result number is claimed.
The only executed validations are the static checks and the synthetic CPU smoke tests
above. The pipeline is designed so that a user who downloads the dataset can run the
documented stages end to end; RQ-VAE / RQ-OPQ SID construction is explicitly
delegated to external public repositories.
