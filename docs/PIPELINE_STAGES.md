# PIPELINE STAGES

Where each stage lives, what it calls, what it needs, and whether a reviewer can
run it without the authors' data/checkpoints.

Runnability labels:

* `[runs as-is]` — CPU-only, synthetic inputs, no data or checkpoints needed.
* `[needs data]` — needs your own preprocessed data / SID arrays (`CFLA_ARTIFACT_ROOT`).
* `[needs retraining]` — needs tokenizer/T5 checkpoints; train first.
* `[reference only]` — shipped for completeness; consumes the authors' frozen
  experiment artifacts, which are not distributed.

| # | stage | wrapper | real entry point | needs | label |
|---|---|---|---|---|---|
| 0 | environment check | `scripts/check_environment.sh` | inline | — | `[runs as-is]` |
| 1 | raw → splits + id maps | `scripts/prepare_data.sh --stage interactions` | `src/data_prep/process_interactions.py` | raw Amazon Reviews 2014 | `[needs data]` |
| 1b | item embeddings | `scripts/prepare_data.sh --stage embeddings` | `src/data_prep/make_item_embeddings.py` | raw metadata + sentence-t5-base | `[needs data]` |
| 2 | SID construction | `scripts/build_sid.sh` | `src/sid/generator.py` (spectral), `src/tokenization/paper/train_tokenizer_5plus1.py` (RQ-VAE5), `src/tokenization/paper/train_rq2opq3_5plus1.py` (RQ2-OPQ3), `src/tokenization/legacy_rqvae/generate_dimension_split_sid_v3.py` (even/random) | item embeddings (+ GPU for RQ tokenizers) | `[needs data]` / `[needs retraining]` |
| 3 | T5 training | `scripts/train.sh` | `src/tiger/train.py` (single models), `src/training/train_t5_5plus1.py` (paper 5+1 pool) | splits + SID arrays + GPU | `[needs retraining]` |
| 4 | candidate generation | `scripts/generate_candidates.sh` | `src/tiger/dump_beams.py` | trained checkpoint + SID array + GPU | `[needs retraining]` |
| 5 | evaluation | `scripts/evaluate.sh` | `scripts/evaluate.py` → `src/fusion/common.py` | one beam dump `.npz` | `[needs data]` |
| 6 | CFLA / paper M5 pipeline | `scripts/run_cfla.sh --stage ...` | `src/cfla/paper_pipeline/{score_m5_candidate_nll,build_m5_rq5,build_m5_cfla,measure_inference_complexity,build_codebook_sensitivity}.py` (`build_single_5plus1.py` is run directly, not through the wrapper) | beam dumps, SID arrays, checkpoints, candidate-NLL parquet (authors' frozen artifacts) | `[reference only]` |
| 6b | paper tables/figures | `scripts/run_cfla.sh --stage table1/efficiency_table/codebook_figure` | `src/cfla/paper_results/*.py` | result CSVs produced by stage 6 | `[reference only]` |
| 7 | ablations | `scripts/run_ablations.sh` | `src/cfla/final_ablation/fa_cpu_ablations.py` | the authors' frozen candidate-NLL artifacts | `[reference only]` |
| 8 | other mechanism studies | run their modules directly | `src/cfla/{candidate_conditioned,cross_tokenizer_shared,factorization_generality,multifactor_fusion,sports_confirmation}/` | frozen artifacts | `[reference only]` |
| 9 | smoke test | `scripts/smoke_test.sh` | `examples/synthetic_demo.py`, `tests/run_tests.py` | — | `[runs as-is]` |

Notes.

* Stages 1–4 are the from-scratch reproduction path; they need the dataset and
  GPU retraining of every factorization used by the paper.
* Stages 6–8 are inference-only code that reproduces the paper's numbers from
  the authors' frozen artifacts. They are shipped because they document exactly
  how each number was computed; they cannot be executed by a reviewer without
  those artifacts. These reference-only scripts expect the authors' frozen
  output tree at `<repo>/outputs` (the default `CFLA_OUTPUT_ROOT`); if you set
  `CFLA_OUTPUT_ROOT` elsewhere, place that tree there or run them from the
  default layout.
* No stage ever inserts the ground-truth target into the candidate pool.
  Calibration statistics are fitted on validation candidates only.
