#!/usr/bin/env bash
# Stage 2 -- Semantic-ID construction (tokenization) for one dataset.
#
#   bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 5
#   bash scripts/build_sid.sh --tokenizer rqvae5 --dataset Beauty
#   bash scripts/build_sid.sh --tokenizer rq2_opq3 --dataset Beauty
#   bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty \
#        --method even
#
# tokenizer=spectral        spectral dimension-partition SID, K in {2,3,4,5},
#                           L=K+1, codebook 256       (src/sid/generator.py)
# tokenizer=rqvae5          RQ-VAE, 5 residual levels + 1 disambiguation token
#                           (src/tokenization/paper/train_tokenizer_5plus1.py)
# tokenizer=rq2_opq3        RQ2-OPQ3 + 1 disambiguation token
#                           (src/tokenization/paper/train_rq2opq3_5plus1.py)
# tokenizer=dimension_split Even / Random dimension split, 5 tokens + 1
#                           disambiguation token (legacy audited generator)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"
OUT="${CFLA_OUTPUT_ROOT:-$REPO/outputs}"

usage() { cat <<'USAGE'
Usage: bash scripts/build_sid.sh --tokenizer TOK --dataset NAME [options]

  --tokenizer TOK   spectral | rqvae5 | rq2_opq3 | dimension_split
  --dataset NAME    dataset name (directory under $CFLA_ARTIFACT_ROOT/data/)
  --k K             spectral only: number of dimension groups, 2..5 (required)
  --method M        dimension_split only: even | random (default: even)
  --affinity A      spectral only: nonneg_pearson (canonical) | abs_pearson | cosine
  --codebook-size C slot width (default 256)
  --seed S          random seed (default 2025)
  --pca-dim D       spectral only: per-subspace PCA whitening dim (default 64)
  --device D        rqvae5 only: torch device (default cuda:0)
  --epochs N        rqvae5 only: tokenizer training epochs (default 3000)
  --dry-run         print the underlying command without running it
USAGE
}
TOKENIZER=""; DATASET=""; K=""; METHOD="even"; AFFINITY="nonneg_pearson"
CODEBOOK=256; SEED=2025; PCA_DIM=64; DEVICE="cuda:0"; EPOCHS=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tokenizer) TOKENIZER="${2:?}"; shift 2 ;;
    --dataset) DATASET="${2:?}"; shift 2 ;;
    --k) K="${2:?}"; shift 2 ;;
    --method) METHOD="${2:?}"; shift 2 ;;
    --affinity) AFFINITY="${2:?}"; shift 2 ;;
    --codebook-size) CODEBOOK="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --pca-dim) PCA_DIM="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$TOKENIZER" ] || { echo "error: --tokenizer is required" >&2; exit 2; }
[ -n "$DATASET" ] || { echo "error: --dataset is required" >&2; exit 2; }
run() { if [ "$DRY" = 1 ]; then echo "DRY-RUN: $*"; else "$@"; fi; }

case "$TOKENIZER" in
  spectral)
    [ -n "$K" ] || { echo "error: --k is required for tokenizer=spectral" >&2; exit 2; }
    run "$PY" -m src.sid.generator --dataset "$DATASET" --affinity "$AFFINITY" \
        --k "$K" --codebook_size "$CODEBOOK" --seed "$SEED" --pca_dim "$PCA_DIM" \
        --alignment-mode canonical
    ;;
  rqvae5)
    EXTRA=(); [ -n "$EPOCHS" ] && EXTRA=(--epochs "$EPOCHS")
    run "$PY" src/tokenization/paper/train_tokenizer_5plus1.py --dataset "$DATASET" \
        --arm rqvae5 --codebook "$CODEBOOK" "${EXTRA[@]}" $( [ "$DRY" = 1 ] && echo --dry-run )
    ;;
  rq2_opq3)
    run "$PY" src/tokenization/paper/train_rq2opq3_5plus1.py --dataset "$DATASET" \
        --codebook "$CODEBOOK" $( [ "$DRY" = 1 ] && echo --dry-run )
    ;;
  dimension_split)
    EMB="$ART/data/$DATASET/item_emb.parquet"
    BASE="$OUT/sids/$DATASET/dimension_split/$METHOD/base_sid.npy"
    FINAL="$OUT/sids/$DATASET/dimension_split/$METHOD/sid.npy"
    run mkdir -p "$(dirname "$FINAL")"
    run "$PY" src/tokenization/legacy_rqvae/generate_dimension_split_sid_v3.py \
        --item_emb_path "$EMB" --output_path "$BASE" --num_sid_tokens 5 \
        --split_method "$METHOD" --codebook_size "$CODEBOOK" \
        --pca_dim "$PCA_DIM" --random_state "$SEED"
    run "$PY" src/tokenization/legacy_rqvae/append_collision_token.py \
        --input_path "$BASE" --output_path "$FINAL" --codebook_size "$CODEBOOK"
    ;;
  *) echo "error: unknown --tokenizer $TOKENIZER" >&2; exit 2 ;;
esac
echo "build_sid: DONE ($TOKENIZER, $DATASET${K:+, K=$K})"
