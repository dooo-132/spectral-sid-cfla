#!/usr/bin/env bash
# Stage 3 -- train a T5 recommender.
#
#   bash scripts/train.sh --model t5_single --dataset Beauty --k 5
#   bash scripts/train.sh --model t5_single --dataset Beauty \
#        --code-path /path/to/Beauty_t5_rqvae5.npy --sid-len 6   # external SID
#
# model=t5_single   one T5 model per SID array (python3 -m src.tiger.train).
#                   --k/--affinity pick the canonical spectral SID array; use
#                   --code-path (and optionally --sid-len) for any other SID
#                   system, e.g. the RQ-VAE / RQ-OPQ arrays built with the
#                   external repositories linked in README section 5.
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
Usage: bash scripts/train.sh --model t5_single --dataset NAME [options]

  --model M          t5_single
  --dataset NAME     dataset name
  --k K              spectral K, 2..5 (required unless --code-path is given)
  --affinity A       nonneg_pearson (canonical; default)
  --sid-len L        SID length; default K+1 for spectral, inferred otherwise
  --code-path FILE   SID array to train on (default: canonical spectral array)
  --num-workers N    DataLoader workers (default 4; use 0 on restricted hosts)
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
BEAM=30; EVAL_MODE="item"; DRY=0; CODE_OVERRIDE=""; SID_LEN=""; NUM_WORKERS=""
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
    --sid-len) SID_LEN="${2:?}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
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
    if [ -n "$CODE_OVERRIDE" ] && [ -z "$K" ]; then
      echo "error: pass --k (spectral) or both --code-path and --sid-len" >&2
    fi
    if [ -n "$SID_LEN" ]; then L="$SID_LEN"; else L=$(( ${K:-0} + 1 )); fi
    [ "$L" -ge 1 ] || { echo "error: cannot determine SID length; pass --sid-len" >&2; exit 2; }
    VOCAB=$((256 * L + 1))
    CODE="${CODE_OVERRIDE:-$ART/data/$DATASET/${DATASET}_t5_spectral_${AFFINITY}_v3_k${K}_sid${L}.npy}"
    TAG="${DATASET}_k${K:-custom}_sid${L}"
    SAVE="$OUT/runs/t5_${TAG}_seed${SEED}.pth"
    LOG="$OUT/runs/t5_${TAG}_seed${SEED}.log"
    run "$PY" -m src.tiger.train --dataset_path "$ART/data/$DATASET" \
        --code_path "$CODE" --vocab_size "$VOCAB" --codebook_size "$CODEBOOK" \
        --max_len 20 --batch_size "$BATCH" --lr "$LR" --num_epochs "$EPOCHS" \
        --early_stop 10 --infer_size 96 --beam_size "$BEAM" --device "$DEVICE" \
        --seed "$SEED" --eval_mode "$EVAL_MODE" --mode "$MODE" \
        --log_path "$LOG" --save_path "$SAVE" ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"}
    ;;
  *) echo "error: unknown --model $MODEL" >&2; exit 2 ;;
esac
echo "train: DONE ($MODEL, $DATASET)"
