#!/usr/bin/env bash
# Stage 2 -- the paper-owned Semantic-ID constructions built in this repository.
#
#   bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 5
#   bash scripts/build_sid.sh --tokenizer dimension_split --dataset Beauty --method even
#
# tokenizer=spectral        dimension-dimension Pearson affinity -> spectral
#                           partition -> per-subspace k-means -> disambiguation
#                           token (src/sid/), K in {2,3,4,5}, L = K+1.
# tokenizer=dimension_split Even / Random dimension split, 5 semantic tokens +
#                           1 disambiguation token (src/tokenization/dimension_split/).
#
# RQ-VAE and RQ-OPQ tokenizers are third-party/derived implementations and are
# deliberately NOT bundled in this release.  Build those SID arrays with the
# external repositories linked in README section 5, then pass the resulting
# `<Dataset>_t5_rqvae*.npy` to scripts/train.sh / generate_candidates.sh.
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

  --tokenizer TOK   spectral | dimension_split
  --dataset NAME    dataset name (directory under $CFLA_ARTIFACT_ROOT/data/)
  --k K             spectral only: number of dimension groups, 2..5 (required)
  --method M        dimension_split only: even | random (default: even)
  --affinity A      spectral only: nonneg_pearson (canonical) | abs_pearson | cosine
  --codebook-size C slot width (default 256)
  --seed S          random seed (default 2025)
  --pca-dim D       per-subspace PCA whitening dim (default 64)
  --dry-run         print the underlying command without running it
USAGE
}
TOKENIZER=""; DATASET=""; K=""; METHOD="even"; AFFINITY="nonneg_pearson"
CODEBOOK=256; SEED=2025; PCA_DIM=64; DRY=0
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
  dimension_split)
    EMB="$ART/data/$DATASET/item_emb.parquet"
    BASE="$OUT/sids/$DATASET/dimension_split/$METHOD/base_sid.npy"
    FINAL="$OUT/sids/$DATASET/dimension_split/$METHOD/sid.npy"
    run mkdir -p "$(dirname "$FINAL")"
    run "$PY" -m src.tokenization.dimension_split.generate_dimension_split_sid_v3 \
        --item_emb_path "$EMB" --output_path "$BASE" --num_sid_tokens 5 \
        --split_method "$METHOD" --codebook_size "$CODEBOOK" \
        --pca_dim "$PCA_DIM" --random_state "$SEED"
    run "$PY" -m src.tokenization.dimension_split.append_collision_token \
        --input_path "$BASE" --output_path "$FINAL" --codebook_size "$CODEBOOK"
    ;;
  *) echo "error: unknown --tokenizer $TOKENIZER (spectral | dimension_split);" >&2
     echo "       RQ-VAE/RQ-OPQ SIDs are built with the external repos in README section 5." >&2
     exit 2 ;;
esac
echo "build_sid: DONE ($TOKENIZER, $DATASET${K:+, K=$K})"
