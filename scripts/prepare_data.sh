#!/usr/bin/env bash
# Stage 1 -- data preparation: raw Amazon Reviews 2014 -> preprocessed splits,
# id mappings and (optionally) sentence-t5-base item embeddings.
#
#   bash scripts/prepare_data.sh --stage all --dataset Beauty --raw-dir /data/raw
#
# Requires the official Amazon Reviews 2014 files (never shipped):
#   <raw-dir>/reviews_<Dataset>_5.json.gz
#   <raw-dir>/meta_<Dataset>.json.gz
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/prepare_data.sh --dataset <name> [options]

  --dataset NAME        Beauty | Cell_Phones_and_Accessories |
                        Health_and_Personal_Care | Toys_and_Games |
                        Sports_and_Outdoors
  --stage STAGE         interactions | embeddings | all   (default: all)
  --raw-dir DIR         directory holding the raw Amazon 2014 files
                        (default: $CFLA_RAW_DATA or <repo>/raw)
  --out-dir DIR         output dataset directory
                        (default: $CFLA_ARTIFACT_ROOT/data/<NAME>)
  --model-dir DIR       local sentence-t5-base checkpoint for --stage embeddings
                        (default: $CFLA_SENTENCE_T5_DIR or <repo>/sentence-t5-base)
  --synthetic-smoke     use a tiny SYNTHETIC reviews file instead of the real
                        dataset (smoke-testing only; labelled synthetic)

Writes: <out-dir>/{<NAME>.json,user_mapping.npy,item_mapping.npy,
                   train.parquet,valid.parquet,test.parquet[,item_emb.parquet]}
USAGE
}

DATASET=""; STAGE="all"; RAW_DIR="${CFLA_RAW_DATA:-$REPO/raw}"; OUT_DIR=""
MODEL_DIR="${CFLA_SENTENCE_T5_DIR:-$REPO/sentence-t5-base}"; SYNTH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:?}"; shift 2 ;;
    --stage) STAGE="${2:?}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?}"; shift 2 ;;
    --synthetic-smoke) SYNTH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$DATASET" ] || { echo "error: --dataset is required" >&2; exit 2; }
case "$STAGE" in interactions|embeddings|all) ;; *) echo "error: bad --stage $STAGE" >&2; exit 2 ;; esac

OUT_ARGS=()
[ -n "$OUT_DIR" ] && OUT_ARGS=(--out-dir "$OUT_DIR")
SMOKE_ARGS=()
[ "$SYNTH" = 1 ] && SMOKE_ARGS=(--synthetic-smoke)

if [ "$STAGE" = interactions ] || [ "$STAGE" = all ]; then
  echo "== [1/2] interactions + splits: $DATASET =="
  "$PY" src/data_prep/process_interactions.py --dataset "$DATASET" \
      --raw-dir "$RAW_DIR" "${OUT_ARGS[@]}" "${SMOKE_ARGS[@]}"
fi
if [ "$STAGE" = embeddings ] || [ "$STAGE" = all ]; then
  echo "== [2/2] item embeddings: $DATASET =="
  "$PY" src/data_prep/make_item_embeddings.py --dataset "$DATASET" \
      --raw-dir "$RAW_DIR" --model-dir "$MODEL_DIR" "${OUT_ARGS[@]}" \
      "${SMOKE_ARGS[@]}"
fi
echo "prepare_data: DONE ($DATASET, stage=$STAGE)"
