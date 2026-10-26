#!/usr/bin/env bash
# Stage 6 -- CFLA: candidate union -> teacher-forced candidate NLL ->
# validation-fitted calibration -> log-sum-exp aggregation -> evaluation.
#
#   bash scripts/run_cfla.sh --config configs/cfla_pool.example.yaml --check
#   bash scripts/run_cfla.sh --config configs/cfla_pool.example.yaml --device cuda
#
# The config lists, for every factorization, the SID array, the trained T5
# checkpoint and the valid/test beam dumps.  Build them with:
#   prepare_data.sh -> build_sid.sh -> train.sh -> generate_candidates.sh
# (RQ-VAE / RQ-OPQ SID arrays are built with the external repositories linked
# in README section 5.)
#
# Real entry point: `python3 -m src.cfla.run`.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"

usage() { cat <<'USAGE'
Usage: bash scripts/run_cfla.sh --config FILE [options]

  --config FILE     CFLA pool config YAML (see configs/cfla_pool.example.yaml)
  --check           only validate that every input path exists
  --device D        cpu (default) | cuda
  --batch-size N    rows per teacher-forcing batch (default 256)
  --methods LIST    cfla,uniform,minnll (default: all)
  --bootstrap N     paired user-level bootstrap resamples (default 5000; 0 = off)
  --output-dir DIR  output directory (default from config / <repo>/outputs/cfla)
  --overwrite       allow writing into a non-empty output directory

Environment:
  CFLA_ARTIFACT_ROOT / CFLA_OUTPUT_ROOT are expanded inside the config.
USAGE
}
CONFIG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done
[ -n "$CONFIG" ] || { echo "error: --config is required" >&2; usage >&2; exit 2; }

echo "run_cfla -> python3 -m src.cfla.run --config $CONFIG $*"
"$PY" -m src.cfla.run --config "$CONFIG" "$@"
echo "run_cfla: DONE"
