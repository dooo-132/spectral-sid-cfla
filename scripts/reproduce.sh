#!/usr/bin/env bash
# Orchestrate the reproduction pipeline stage by stage.
#
#   bash scripts/reproduce.sh --stage all          # from-scratch (needs data + GPU)
#   bash scripts/reproduce.sh --stage data --dataset Beauty
#   bash scripts/reproduce.sh --stage all --dry-run
#
# Stages (in pipeline order):
#   data      raw Amazon Reviews 2014 -> splits, id maps, item embeddings
#   sid       SID construction for every factorization
#   train     T5 training (one model per factorization)
#   generate  beam-search candidate dumps
#   evaluate  item-level Recall@K / NDCG@K of one dump
#   cfla      CFLA / M5 pipeline over frozen artifacts (reference-only stages)
#   tables    paper table/figure builders (reference-only stages)
#
# This script does NOT ship or download data and cannot train the paper's models
# by itself: a full run needs the official Amazon Reviews 2014 files and a GPU.
# Every stage is a thin wrapper around the individual scripts; use them directly
# with extra flags if you need finer control.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"
OUT="${CFLA_OUTPUT_ROOT:-$REPO/outputs}"

usage() { cat <<'USAGE'
Usage: bash scripts/reproduce.sh --stage STAGE [options]

  --stage STAGE      data | sid | train | generate | evaluate | cfla | tables | all
  --dataset NAME     dataset (default: Beauty)
  --raw-dir DIR      raw Amazon Reviews 2014 directory (stage data)
  --dry-run          print the wrapped commands instead of running them
  -h, --help         this message

Environment:
  CFLA_ARTIFACT_ROOT  input artifact root (default <repo>/artifacts)
  CFLA_OUTPUT_ROOT    output root         (default <repo>/outputs)
  CFLA_RAW_DATA       raw dataset dir     (default <repo>/raw)
USAGE
}
STAGE=""; DATASET="Beauty"; RAW_DIR="${CFLA_RAW_DATA:-$REPO/raw}"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?}"; shift 2 ;;
    --dataset) DATASET="${2:?}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$STAGE" ] || { echo "error: --stage is required" >&2; usage >&2; exit 2; }

run() {
  if [ "$DRY" = 1 ]; then echo "DRY-RUN: $*"; else "$@"; fi
}

check_artifacts() {
  if [ ! -d "$ART" ]; then
    echo "FATAL: CFLA_ARTIFACT_ROOT does not exist: $ART" >&2
    echo "Prepare the input artifacts first (docs/DATA_FORMAT.md) or set" >&2
    echo "CFLA_ARTIFACT_ROOT; stage '$STAGE' reads them." >&2
    exit 2
  fi
}

stage_data() {
  echo "== stage data: $DATASET =="
  run bash scripts/prepare_data.sh --dataset "$DATASET" --stage all \
      --raw-dir "$RAW_DIR"
}

stage_sid() {
  echo "== stage sid: spectral K=2..5 + RQ-VAE5 + RQ2-OPQ3 + even/random split =="
  for K in 2 3 4 5; do
    run bash scripts/build_sid.sh --tokenizer spectral --dataset "$DATASET" --k "$K"
  done
  run bash scripts/build_sid.sh --tokenizer rqvae5 --dataset "$DATASET"
  run bash scripts/build_sid.sh --tokenizer rq2_opq3 --dataset "$DATASET"
  run bash scripts/build_sid.sh --tokenizer dimension_split --dataset "$DATASET" --method even
  run bash scripts/build_sid.sh --tokenizer dimension_split --dataset "$DATASET" --method random
}

stage_train() {
  echo "== stage train: one T5 per factorization (needs a GPU) =="
  echo "   SID arrays are the ones built by scripts/build_sid.sh under CFLA_OUTPUT_ROOT."
  for K in 2 3 4 5; do
    run bash scripts/train.sh --model t5_single --dataset "$DATASET" --k "$K" \
        --code-path "$OUT/sids/$DATASET/nonneg_pearson/k$K/seed2025/sid.npy"
  done
  run bash scripts/train.sh --model t5_5plus1 --dataset "$DATASET" --arm rqvae5
}

stage_generate() {
  echo "== stage generate: beam-30 candidate dumps for every spectral model =="
  for K in 2 3 4 5; do
    for SPLIT in valid test; do
      run bash scripts/generate_candidates.sh --dataset "$DATASET" --k "$K" \
          --split "$SPLIT" \
          --code-path "$OUT/sids/$DATASET/nonneg_pearson/k$K/seed2025/sid.npy"
    done
  done
}

stage_evaluate() {
  echo "== stage evaluate: item-level metrics of the K=5 test dump =="
  DUMP="$OUT/beam_dumps/$DATASET/test_k5_sid6.npz"
  run bash scripts/evaluate.sh --dump "$DUMP" \
      --out "$OUT/metrics/eval_${DATASET}_k5_test.json"
}

stage_cfla() {
  echo "== stage cfla: CFLA / M5 pipeline over the authors' frozen artifacts =="
  echo "   NOTE: these stages consume frozen experiment artifacts and are not"
  echo "   reproducible from a fresh clone without them (docs/PIPELINE_STAGES.md)."
  check_artifacts
  run bash scripts/run_cfla.sh --stage candidate_nll --datasets "$DATASET" --dry-run
  run bash scripts/run_cfla.sh --stage m5 --datasets "$DATASET" --dry-run
  run bash scripts/run_cfla.sh --stage m5_cfla --datasets "$DATASET" --dry-run
}

stage_tables() {
  echo "== stage tables: paper table/figure builders (from frozen result CSVs) =="
  run bash scripts/run_cfla.sh --stage table1
  run bash scripts/run_cfla.sh --stage efficiency_table
  run bash scripts/run_cfla.sh --stage codebook_figure
}

case "$STAGE" in
  data) stage_data ;;
  sid) stage_sid ;;
  train) stage_train ;;
  generate) stage_generate ;;
  evaluate) stage_evaluate ;;
  cfla) stage_cfla ;;
  tables) stage_tables ;;
  all)
    stage_data; stage_sid; stage_train; stage_generate; stage_evaluate
    echo; echo "== from-scratch path finished; CFLA/tables stages are reference-only =="
    stage_cfla; stage_tables
    ;;
  *) echo "error: unknown --stage $STAGE" >&2; usage >&2; exit 2 ;;
esac
echo "reproduce: DONE (stage=$STAGE, dataset=$DATASET)"
