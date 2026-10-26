#!/usr/bin/env bash
# Stage 4 -- generate beam candidates (top-30 by default) for one trained model
# and one split, writing a beam dump in the format documented in
# docs/DATA_FORMAT.md.
#
#   bash scripts/generate_candidates.sh --dataset Beauty --k 2 \
#        --ckpt-path outputs/runs/t5_Beauty_nonneg_pearson_k2_sid3_seed2025.pth
#
# Real entry point: python3 -m src.tiger.dump_beams
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"
OUT="${CFLA_OUTPUT_ROOT:-$REPO/outputs}"

usage() { cat <<'USAGE'
Usage: bash scripts/generate_candidates.sh --dataset NAME --k K [options]

  --dataset NAME     dataset name (required)
  --k K              spectral K of the trained model, 2..5 (required)
  --affinity A       nonneg_pearson (canonical; default)
  --ckpt-path FILE   trained checkpoint (default: <outputs>/runs/t5_<...>.pth)
  --code-path FILE   SID array (default: <artifacts>/data/<name>/..._sid<L>.npy;
                     pass the output of scripts/build_sid.sh here)
  --output-path FILE beam dump .npz (default: <outputs>/beam_dumps/<name>/<split>_k<K>_sid<L>.npz)
  --split S          valid | test (default: test)
  --beam-size N      default 30
  --infer-size N     default 96
  --max-len N        default 20
  --codebook-size C  default 256
  --device D         cuda (default) | cpu
  --seed S           default 2025
  --dry-run          print the underlying command without running it
USAGE
}
DATASET=""; K=""; AFFINITY="nonneg_pearson"; CKPT=""; CODE=""; OUTP=""
SPLIT="test"; BEAM=30; INFER=96; MAXLEN=20; CODEBOOK=256; DEVICE="cuda"; SEED=2025; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:?}"; shift 2 ;;
    --k) K="${2:?}"; shift 2 ;;
    --affinity) AFFINITY="${2:?}"; shift 2 ;;
    --ckpt-path) CKPT="${2:?}"; shift 2 ;;
    --code-path) CODE="${2:?}"; shift 2 ;;
    --output-path) OUTP="${2:?}"; shift 2 ;;
    --split) SPLIT="${2:?}"; shift 2 ;;
    --beam-size) BEAM="${2:?}"; shift 2 ;;
    --infer-size) INFER="${2:?}"; shift 2 ;;
    --max-len) MAXLEN="${2:?}"; shift 2 ;;
    --codebook-size) CODEBOOK="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$DATASET" ] || { echo "error: --dataset is required" >&2; exit 2; }
[ -n "$K" ] || { echo "error: --k is required" >&2; exit 2; }
L=$((K + 1)); TAG="${DATASET}_${AFFINITY}_k${K}_sid${L}"
[ -n "$CODE" ] || CODE="$ART/data/$DATASET/${DATASET}_t5_spectral_${AFFINITY}_v3_k${K}_sid${L}.npy"
[ -n "$CKPT" ] || CKPT="$OUT/runs/t5_${TAG}_seed${SEED}.pth"
[ -n "$OUTP" ] || OUTP="$OUT/beam_dumps/$DATASET/${SPLIT}_k${K}_sid${L}.npz"
mkdir -p "$(dirname "$OUTP")"
CMD=("$PY" -m src.tiger.dump_beams --dataset_path "$ART/data/$DATASET" \
     --code_path "$CODE" --ckpt_path "$CKPT" --output_path "$OUTP" \
     --split "$SPLIT" --sid_len "$L" --codebook_size "$CODEBOOK" \
     --beam_size "$BEAM" --infer_size "$INFER" --max_len "$MAXLEN" \
     --device "$DEVICE" --seed "$SEED")
if [ "$DRY" = 1 ]; then echo "DRY-RUN: ${CMD[*]}"; else "${CMD[@]}"; fi
echo "generate_candidates: DONE ($DATASET K=$K split=$SPLIT -> $OUTP)"
