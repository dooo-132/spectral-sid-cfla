# PIPELINE STAGES

Where each stage lives, what it calls, what it needs, and whether it can run
without a GPU.

| # | stage | wrapper | real entry point | inputs | notes |
|---|---|---|---|---|---|
| 0 | environment check | `scripts/check_environment.sh` | inline | — | prints versions and resolved path roots |
| 1 | raw -> splits + id maps | `scripts/prepare_data.sh --stage interactions` | `src/data_prep/process_interactions.py` | Amazon Reviews 2014 (`reviews_<D>_5.json.gz`) | 1-based ids, leave-one-out splits |
| 1b | item embeddings | `scripts/prepare_data.sh --stage embeddings` | `src/data_prep/make_item_embeddings.py` | `meta_<D>.json.gz` + sentence-t5-base | 768-d `item_emb.parquet` |
| 2 | SID construction (paper-owned) | `scripts/build_sid.sh` | `src/sid/generator.py` (spectral), `src/tokenization/dimension_split/` (even/random) | item embeddings | K=2..5 or dimension split; injective SIDs |
| 2b | SID construction (external) | — | external RQ-VAE / RQ-OPQ repositories (README §5.2–5.3) | item embeddings | produce the same `.npy` format; not bundled |
| 3 | T5 training | `scripts/train.sh` | `src/tiger/train.py` | splits + one SID array per factorization | GPU for real runs; CPU works for small data |
| 4 | candidate generation | `scripts/generate_candidates.sh` | `src/tiger/dump_beams.py` | trained checkpoint + SID array | beam dumps for valid/test, 0-based ids |
| 5 | evaluation of one dump | `scripts/evaluate.sh` | `scripts/evaluate.py` -> `src/fusion/common.py` | one beam dump | Recall@K / NDCG@K |
| 6 | CFLA | `scripts/run_cfla.sh` | `src/cfla/run.py` (`src/cfla/cfla.py`) | all beam dumps + SID arrays + checkpoints + parquet splits | union -> candidate NLL -> calibration -> LSE -> metrics + bootstrap |
| 7 | full from-scratch run | `scripts/reproduce.sh` | wraps stages 1-6 | dataset + GPU | one factorization set per run |
| 8 | smoke tests | `scripts/smoke_test.sh`, `scripts/smoke_pipeline.sh` | inline / whole pipeline | none (synthetic, CPU) | env + compile + imports, and a full synthetic end-to-end run |

Notes.

* `scripts/prepare_data.sh --synthetic-smoke` generates a small **synthetic**
  corpus (labelled synthetic) so the data/SID stages can be exercised without the
  real dataset; `scripts/smoke_pipeline.sh` uses it for the end-to-end CPU check.
* `scripts/build_sid.sh --tokenizer spectral` writes
  `${CFLA_OUTPUT_ROOT}/sids/<Dataset>/<affinity>/k<K>/seed<S>/sid.npy`;
  `scripts/train.sh --code-path` and `scripts/generate_candidates.sh --code-path`
  point at that file (or at an externally built RQ SID array).
* CFLA never inserts the ground-truth target into the candidate pool; the target is
  used only for metric computation after scoring, and the calibration statistics are
  fitted on validation candidates only.
* No stage writes into `CFLA_ARTIFACT_ROOT`.
