# RELEASE VALIDATION

Date: 2026-10-26. Release: `<release_20261026>/spectral-sid-cfla`.
This document distinguishes **executed**, **static-checked** and
**not executed** items. No model training and no full experiment was run.

## 1. Environment used for validation

| item | value |
|---|---|
| Python | 3.12.3 (conda distribution) |
| PyTorch | 2.12.0+cu130 |
| CUDA | 13.0 (GPU present on the authors' machine; **all executed checks below are CPU-only**) |
| Transformers | 5.9.0 |
| scikit-learn / SciPy / NumPy / pandas / pyarrow | 1.9.0 / 1.17.1 / 2.4.6 / 3.0.3 / 24.0.0 |
| PyYAML / tqdm / matplotlib | 6.0.2 / 4.67.3 / 3.9.2 |
| optional: faiss-cpu / sentence-transformers | 1.14.2 / 5.5.1 |

All versions in `requirements.txt` were checked against the installed
environment (12 pins + 2 optional extras; 0 mismatches).

## 2. Level 1 — static validation (executed)

| check | command | result |
|---|---|---|
| Python syntax, whole tree | in-memory `compile()` over all `.py` | **191 files, 0 failures** |
| Shell syntax | `bash -n` over all `scripts/*.sh` | **10 files, 0 failures** |
| YAML syntax | `yaml.safe_load` over all `configs/*.yaml` | **18 files, 0 failures** |
| Import graph | `importlib.import_module` over all 161 non-`__init__` modules under `src/`, `PYTHONPATH=<release>` | **160 OK, 1 expected failure** (see below) |
| Missing third-party modules | import sweep above; every third-party package used is declared in `requirements.txt` | OK |
| Entry points exist | every script/entry point referenced by `scripts/*.sh`, `README.md`, `docs/PIPELINE_STAGES.md` resolved with `find`/`test -f` | OK |
| README file references | every file path in README code blocks (`docs/…`, `scripts/…`, `configs/…`, `src/…`, `tests/…`, `examples/…`) | **14/14 exist** |
| Path independence | scan for POSIX root/home prefixes and the Windows user-profile prefix | **0 hits** |
| Config parameter names | `configs/experiments/*.yaml` parsed by `src.utils.experiment_config`; the runner validates the schema and fails loudly on a missing dataset (by design) | OK |
| No inherited Git history | `find . -name .git` | absent |

Expected import failure: `src.tokenization.legacy_rqvae.generate_code` is the
upstream **script-style** RQ-VAE encoder (it parses arguments and loads a
checkpoint at import time, exactly as upstream). It is invoked as a script by
`src/tokenization/paper/train_tokenizer_5plus1.py` (which writes a
path-patched copy and runs it) and is not meant to be imported. All reference
implementations that *are* importable import cleanly.

Notes.

* `python3 -m src.runner --config configs/experiments/… --dry-run` performs
  config validation **and checks that the dataset directory exists under
  `CFLA_ARTIFACT_ROOT`**; without the user's data it stops with
  `invalid config: experiment.dataset 'Beauty' not found under …/artifacts/data`.
  This is the intended fail-loud behaviour, not a crash.
* `scripts/run_cfla.sh` / `scripts/run_ablations.sh` require the authors' frozen
  artifacts; without `CFLA_ARTIFACT_ROOT` they exit non-zero with an explicit
  message (verified). They are labelled `[reference only]` in the README.

## 3. Level 2 — lightweight smoke tests (executed, CPU, synthetic only)

Command: `bash scripts/smoke_test.sh` → **PASSED**.

| # | test | evidence |
|---|---|---|
| L2-1 | environment check (`scripts/check_environment.sh`) | prints interpreter and resolved `CFLA_ARTIFACT_ROOT`/`CFLA_OUTPUT_ROOT`; passes |
| L2-2 | spectral SID construction, synthetic correlated embeddings (300 items × 24 dims, K=2/3/5) | unit test `tests/test_sid_construction.py`: shape `(N, K+1)`, semantic prefix preserved, **injective final SID**, deterministic with the same seed, written `manifest.sha256` verified |
| L2-3 | collision / disambiguation token | unit test: colliding base SIDs become unique after appending the disambiguation token, prefix unchanged |
| L2-4 | CFLA math: validation-fitted zNLL + log-sum-exp | unit test `tests/test_cfla_math.py`: released `fit_scalers`/`z_of`/`score_lse_from_znll` match an **independent NumPy re-implementation** to `1e-12`; calibration statistics equal the validation-candidate mean/std; score is monotone in each factorization's NLL; changing the target does not change the scores (target is not used by the scoring path) |
| L2-5 | ranking metrics | unit test `tests/test_metrics.py`: Recall@K/NDCG@K and item-id canonicalisation match hand-computed values; invalid/absent targets contribute 0; two-model weight fusion checked manually |
| L2-6 | end-to-end synthetic demo (`examples/synthetic_demo.py`) | builds synthetic item embeddings → spectral SIDs (K=2,3,4, injective) → synthetic per-factorization NLLs → validation-fitted zNLL → LSE aggregation, and asserts the released LSE equals an independent reference exactly; prints SYNTHETIC ranking metrics (explicitly labelled not paper results) |
| L2-7 | unit-test suite | `tests/run_tests.py`: **13 tests, OK** (≈25 s, CPU) |
| L2-8 | from-scratch data/SID path (synthetic raw data) | `scripts/prepare_data.sh --dataset Beauty --stage all --synthetic-smoke` → 400 synthetic items, splits + `item_mapping.npy` + `item_emb.parquet`; then `scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 2..5` → `400/400` unique SIDs, injective, metadata + manifest written; `--tokenizer dimension_split --method even/random` → `400/400` unique SIDs |
| L2-9 | evaluation script on a synthetic beam dump | `scripts/evaluate.sh --dump <synthetic .npz>` → prints item-level Recall@5/10/20 and NDCG@5/10/20 (finite, in `[0,1]`), `n_users=20`, `target_in_union_rate=1.0` |

All synthetic inputs are generated at run time with fixed seeds inside
temporary directories; **no real experiment artifact is used** and no artifact
is written into the release tree.

## 4. Level 3 — release reproducibility audit

### 4.1 README commands

README contains 22 executable command lines in 10 fenced blocks across its 11 sections. Every referenced script,
config, docs file and example path exists (checked above). Classification of the
commands:

| class | count | notes |
|---|---|---|
| `[runs as-is]` (CPU, synthetic, no data) | 4 | `check_environment.sh`, `smoke_test.sh`, `examples/synthetic_demo.py`, `run_ablations.sh --list` (the unit-test runner is documented separately in §3) |
| `[needs your data]` | 8 | `prepare_data.sh` (raw Amazon Reviews 2014 + sentence-t5-base), `build_sid.sh --tokenizer spectral/rq2_opq3/dimension_split`, `evaluate.sh`, `src.sid.generator` |
| `[needs retraining]` | 9 | `train.sh` (t5_single / t5_5plus1), `generate_candidates.sh`, `build_sid.sh --tokenizer rqvae5` |
| `[reference only]` | 3 | `run_cfla.sh --stage {candidate_nll,m5,m5_cfla}`, `run_ablations.sh --stage final-ablation`, paper table/figure builders |

Executed subset: the 6 `[runs as-is]` commands (all pass) plus the synthetic-raw
variant of the data/SID pipeline (L2-8) and the synthetic-dump variant of
`evaluate` (L2-9). The `[needs your data]` / `[needs retraining]` commands were
**not executed** (no dataset, no GPU training in this release task); their
existence, flags and input contracts were verified statically against the real
`argparse` definitions. The `[reference only]` commands were executed once to
confirm they fail loudly with an explicit "missing frozen artifact" error rather
than silently — verified for `run_cfla.sh --stage candidate_nll` (device/artifact
guard) and `--stage table1` (missing result CSV), and for `run_ablations.sh`
(missing candidate-NLL parquet).

### 4.2 What was NOT executed and why

| item | why |
|---|---|
| T5 training (24 single models + 5 pool arms) | requires the Amazon Reviews 2014 data and a GPU; explicitly out of scope for this release task |
| RQ-VAE5 / RQ2-OPQ3 tokenizer training | same (GPU + data) |
| beam candidate generation / teacher-forced NLL scoring at scale | requires trained checkpoints and the dataset |
| paper Table 1 / efficiency / codebook reproduction | requires the authors' frozen artifacts (not distributed) or a full retraining |
| any claim of reproducing the paper's numbers | **never made**: no results are shipped and no experiment was run here |

### 4.3 Environment / dependency audit

`requirements.txt` was installed-version-checked against the validation
environment (0 mismatches). `torch==2.12.0` is the CPU wheel by default; the
CUDA-13 build actually used by the authors is documented in `docs/ENVIRONMENT.md`
and `README.md` §3. `faiss-cpu` and `sentence-transformers` are optional and
only needed by the RQ-OPQ tokenizer and the item-embedding preparation stage
respectively.

### 4.4 Defects found during validation and fixed

1. `src/tiger/dump_beams.py` imported `main`/`dataset`/`dataloader` as flat
   modules (broken outside the original directory layout) → replaced with
   package-relative imports plus a script-mode fallback.
2. `src/data_prep/make_item_embeddings.py` was still the original notebook-style
   script (hardcoded dataset, no CLI) while `scripts/prepare_data.sh` called it
   with flags → rewritten as a CLI with the processing logic unchanged, plus a
   labelled `--synthetic-smoke` path.
3. `src/cfla/candidate_conditioned/cc_common.py` missed the
   `factorization_generality/` directory on its import path → fixed.
4. Reference RQ-VAE modules used flat sibling imports (`datasets`, `utils`) that
   only worked when run from their own directory → explicit sibling-path shim
   added (no logic change).
5. `src/legacy_audited/phase5_fusion.py` created a directory at import time and
   imported the historical fusion module by a name that no longer exists → the
   import-time write was removed and the import repointed to the shipped
   compatibility shim (`src/legacy_audited/fusion_audit.py`).
6. Paper-pipeline / training / table-builder scripts contained hardcoded
   authors' paths and internal output directories → replaced with the
   environment-variable roots and release-relative paths.
7. Internal workflow references ("prompt section N", "PHASE N", internal report
   names) and Chinese-language comments → removed / translated (see
   `docs/ANONYMITY_AUDIT.md`).
8. Docs referencing dropped internal reports → references removed and the
   technical content folded into `docs/DATA_FORMAT.md` /
   `docs/SID_REPRODUCTION_REPORT.md` / `docs/EXPERIMENT_PROTOCOL.md`.
9. `.gitignore` had an unanchored `paper/` rule that silently excluded the
   RQ-VAE5 tokenizer code (`src/tokenization/paper/`) and the frozen tokenizer
   recipe (`configs/paper/`) from the git index, i.e. exactly the paper-final
   tokenization code would have been missing from the upload. The rule is now
   anchored to `/paper/`; the git index was re-checked and contains all 236
   files.

## 5. Independent cross-review

Two independent read-only reviews were commissioned on the assembled tree (a
reviewer that is not the release author): one during assembly and one on the
next frozen commit. Both were adversarial (assume leaks remain).

**Review 1 (during assembly).** The reviewer found P0 anonymity leaks that the
author's earlier scans had missed: the real GitHub owner handle and repository
URL had been written into shipped docs, absolute local paths and internal
project/audit/branch/commit identifiers remained in several docs, source files
and one config, and a dated internal checkpoint path and commit hash remained in
two source files. All of them were fixed, and the affected docs were rewritten
so that their own descriptions do not reproduce the sensitive strings.

**Review 2 (the intermediate pre-fix tree, 236 tracked files).**

| cross-review item | result |
|---|---|
| tree completeness vs README | PASS |
| README command audit (files/flags/classification) | CONCERN: all referenced files and flags existed, but the documented from-scratch SID→train path layout did not chain, paper table builders resolved inputs under a hard-coded location, `run_cfla.sh --check` was unreachable behind the artifact guard, and `--stage single` ignored its flags |
| independent anonymity scan | CONCERN: the working tree and HEAD were clean, but `.git` still held an unreachable blob with the pre-fix docs (containing the real owner handle), and generic internal workflow markers remained despite the audit's removal claim |
| runnable checks re-run independently | PASS (`13 tests OK`; synthetic-demo assertions pass) |
| honesty audit | CONCERN: some config files embedded unverifiable result assertions; two test files cited by `docs/CODE_MAPPING.md` were not shipped; line references in `CODE_MAPPING` had drifted |
| `docs/CODE_MAPPING.md` sample rows vs tree | FAIL: 13/24 sampled line references stale and several referenced files absent |

**Review verdict on that intermediate tree: NOT READY.** Fixes applied afterwards:

1. `.git` was rebuilt from the clean working tree (delete + re-init + single
   anonymous commit); unreachable objects and the stale blobs are gone;
2. all `__pycache__`/`.pyc` were deleted; the tree is exactly 236 tracked files;
3. `RELEASE_VALIDATION.md` §5 (this section) was replaced with the real review
   history — the earlier pre-emptive "PASS" text was wrong and has been removed;
4. `CODE_MAPPING` was corrected: non-existent test/analysis files removed,
   `process.py` → `process_interactions.py`, and a caveat added that `:line`
   suffixes are approximate while file/function names are authoritative;
5. from-scratch chain: `scripts/train.sh` and `scripts/generate_candidates.sh`
   accept `--code-path`, and `scripts/reproduce.sh` passes the SID arrays built
   by `scripts/build_sid.sh` to training/candidate generation;
6. paper table/figure builders now resolve their inputs through
   `CFLA_OUTPUT_ROOT` and exit with an explicit "frozen result CSVs missing"
   message instead of a raw traceback;
7. `scripts/run_cfla.sh`: the unsupported `single` stage was removed;
   `--check` now passes the artifact-root guard for the checking stages;
8. result assertions were removed from `configs/canonical*.yaml`; the README
   wording was corrected ("no result table or experimental artifact");
9. dangling internal record-name references were neutralized (including one
   remaining in the tokenizer recipe config), and the audit text now states
   precisely what remains: generic workflow wording in reference-only scripts,
   with no identity, institution, server or repository information.

A third independent review was not available before the final freeze, so the
post-fix state was re-checked with the same automated checks in §2–§4 plus the
author's own full-tree scans; the remaining limitation is that the final
independent re-review of the rebuilt commit should be repeated by the authors
before upload (`docs/UPLOAD_CHECKLIST.md` §1). This document does not claim an
independent PASS on the final commit.

## 6. Summary

| level | executed | result |
|---|---|---|
| L1 static | yes | PASS (0 syntax errors, 10/10 shell scripts, 18/18 YAML, 160/161 modules importable, one upstream script-style module intentionally not importable, no absolute paths) |
| L2 smoke | yes | PASS (13 unit tests, synthetic SID construction, independent CFLA-math check, metrics check, synthetic end-to-end demo, synthetic data/SID pipeline, evaluation on a synthetic dump) |
| L3 release audit | partially | PASS for command/file/flag existence and honesty; data-dependent and training-dependent commands not executed (documented above) |

**No training run and no paper-number reproduction was performed or claimed.**
The release is ready for anonymous upload from a validation standpoint, with the
open items listed in `docs/UPLOAD_CHECKLIST.md`.
