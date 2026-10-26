#!/usr/bin/env bash
# Stage 3 -- train a T5 recommender.
#
#   bash scripts/train.sh --model t5_single --dataset Beauty --k 5
#   bash scripts/train.sh --model t5_5plus1 --dataset Beauty --arm rqvae5
#
# model=t5_single   one model per (dataset, K, affinity) SID array; this is the
#                   single-model / per-granularity baseline trainer
#                   (python3 -m src.tiger.train).
# model=t5_5plus1   paper-pool 5+1 tokenizer models (RQ-VAE5 / RQ1-OPQ4 /
#                   OPQ5), L=6, vocabulary inferred from the SID validation
#                   file (src/training/train_t5_5plus1.py).
# Training needs a CUDA GPU; --dry-run is CPU-safe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"
OUT="${CFLA_OUTPUT_ROOT:-$REPO/outputs}"

usage() { cat <<'USAGE'
Usage: bash scripts/train.sh --model {t5_single|t5_5plus1} --dataset NAME [options]

  --model M          t5_single | t5_5plus1
  --dataset NAME     dataset name
  --k K              t5_single: spectral K, 2..5 (required)
  --affinity A       t5_single: nonneg_pearson (canonical; default)
  --arm ARM          t5_5plus1: rqvae5 | rq1_opq4 | opq5 (required)
  --codebook C       token slot width (default 256)
  --device D         cuda (default) | cpu
  --mode M           t5_single: train | evaluation (default: train)
  --seed S           random seed (default 2025)
  --batch-size N     default 256
  --epochs N         default 200
  --lr LR            default 1e-4
  --beam-size N      generation beam for validation (default 30)
  --eval-mode M      item (default, canonical) | token (legacy proxy)
  --code-path FILE   SID array (default: <artifacts>/data/<name>/<...>_sid<L>.npy)
  --dry-run          print the underlying command without running it
USAGE
}
MODEL=""; DATASET=""; K=""; AFFINITY="nonneg_pearson"; ARM=""; CODEBOOK=256
DEVICE="cuda"; MODE="train"; SEED=2025; BATCH=256; EPOCHS=200; LR=1e-4
BEAM=30; EVAL_MODE="item"; DRY=0; CODE_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --dataset) DATASET="${2:?}"; shift 2 ;;
    --k) K="${2:?}"; shift 2 ;;
    --affinity) AFFINITY="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --codebook) CODEBOOK="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --batch-size) BATCH="${2:?}"; shift 2 ;;
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --lr) LR="${2:?}"; shift 2 ;;
    --beam-size) BEAM="${2:?}"; shift 2 ;;
    --eval-mode) EVAL_MODE="${2:?}"; shift 2 ;;
    --code-path) CODE_OVERRIDE="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "error: --model is required" >&2; exit 2; }
[ -n "$DATASET" ] || { echo "error: --dataset is required" >&2; exit 2; }
run() { if [ "$DRY" = 1 ]; then echo "DRY-RUN: $*"; else "$@"; fi; }

case "$MODEL" in
  t5_single)
    [ -n "$K" ] || { echo "error: --k is required for model=t5_single" >&2; exit 2; }
    L=$((K + 1)); VOCAB=$((256 * L + 1))
    CODE="${CODE_OVERRIDE:-$ART/data/$DATASET/${DATASET}_t5_spectral_${AFFINITY}_v3_k${K}_sid${L}.npy}"
    SAVE="$OUT/runs/t5_${DATASET}_${AFFINITY}_k${K}_sid${L}_seed${SEED}.pth"
    LOG="$OUT/runs/t5_${DATASET}_${AFFINITY}_k${K}_sid${L}_seed${SEED}.log"
    run "$PY" -m src.tiger.train --dataset_path "$ART/data/$DATASET" \
        --code_path "$CODE" --vocab_size "$VOCAB" --codebook_size "$CODEBOOK" \
        --max_len 20 --batch_size "$BATCH" --lr "$LR" --num_epochs "$EPOCHS" \
        --early_stop 10 --infer_size 96 --beam_size "$BEAM" --device "$DEVICE" \
        --seed "$SEED" --eval_mode "$EVAL_MODE" --mode "$MODE" \
        --log_path "$LOG" --save_path "$SAVE"
    ;;
  t5_5plus1)
    [ -n "$ARM" ] || { echo "error: --arm is required for model=t5_5plus1" >&2; exit 2; }
    run "$PY" src/training/train_t5_5plus1.py --dataset "$DATASET" --arm "$ARM" \
        --codebook "$CODEBOOK" $( [ "$DRY" = 1 ] && echo --dry-run )
    ;;
  *) echo "error: unknown --model $MODEL" >&2; exit 2 ;;
esac
echo "train: DONE ($MODEL, $DATASET)"
