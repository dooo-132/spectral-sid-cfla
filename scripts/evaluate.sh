#!/usr/bin/env bash
# Stage 5 -- item-level evaluation of a beam dump (Recall@K / NDCG@K,
# K in {5,10,20}).
#
#   bash scripts/evaluate.sh --dump outputs/beam_dumps/Beauty/test_k2_sid3.npz
#
# Real entry point: scripts/evaluate.py over src/fusion/common.py.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"

usage() { cat <<'USAGE'
Usage: bash scripts/evaluate.sh --dump FILE [--out FILE]

  --dump FILE   beam dump .npz produced by scripts/generate_candidates.sh
  --out FILE    optional metrics JSON output path
USAGE
}
DUMP=""; OUTJSON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dump) DUMP="${2:?}"; shift 2 ;;
    --out) OUTJSON="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$DUMP" ] || { echo "error: --dump is required" >&2; exit 2; }
ARGS=(--dump "$DUMP"); [ -n "$OUTJSON" ] && ARGS+=(--out "$OUTJSON")
"$PY" scripts/evaluate.py "${ARGS[@]}"
