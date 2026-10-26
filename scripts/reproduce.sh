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
#   cfla      CFLA scoring over the pools built by the stages above
#
# This script does NOT ship or download data: a full run needs the official
# Amazon Reviews 2014 files (download them yourself, see README section 4) and,
# for training/beam generation, a GPU.  Every stage is a thin wrapper around the
# individual scripts; use them directly with extra flags for finer control.
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

  --stage STAGE      data | sid | train | generate | evaluate | cfla | all
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
  echo "== stage sid: spectral K=2..5 + even/random dimension split =="
  echo "   (RQ-VAE / RQ-OPQ SIDs are built with the external repositories"
  echo "    linked in README section 5 and are not bundled in this release.)"
  for K in 2 3 4 5; do
    run bash scripts/build_sid.sh --tokenizer spectral --dataset "$DATASET" --k "$K"
  done
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
  echo "   NOTE: RQ-VAE / RQ-OPQ factorizations are optional and need SID arrays"
  echo "   produced with the external repositories linked in README section 5;"
  echo "   train them with: bash scripts/train.sh --model t5_single \\"
  echo "       --dataset $DATASET --code-path <sid.npy> --sid-len <L>"
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
  echo "== stage cfla: candidate union -> candidate NLL -> calibration -> LSE -> eval =="
  CFG="${CFLA_CFLA_CONFIG:-configs/cfla_pool.example.yaml}"
  echo "   using CFLA pool config: $CFG (edit it to list the factorizations you built)"
  run bash scripts/run_cfla.sh --config "$CFG" --check
}

case "$STAGE" in
  data) stage_data ;;
  sid) stage_sid ;;
  train) stage_train ;;
  generate) stage_generate ;;
  evaluate) stage_evaluate ;;
  cfla) stage_cfla ;;
  all)
    stage_data; stage_sid; stage_train; stage_generate; stage_evaluate
    echo; echo "== from-scratch path finished; CFLA stages are reference-only =="
    stage_cfla
    ;;
  *) echo "error: unknown --stage $STAGE" >&2; usage >&2; exit 2 ;;
esac
echo "reproduce: DONE (stage=$STAGE, dataset=$DATASET)"
