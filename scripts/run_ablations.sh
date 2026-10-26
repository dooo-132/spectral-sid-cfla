#!/usr/bin/env bash
# Stage 7 -- the paper's candidate-likelihood ablations (CPU-only, frozen
# artifacts; no training).
#
#   bash scripts/run_ablations.sh --study final-ablation --cmd calibration
#   bash scripts/run_ablations.sh --list
#
# Study `final-ablation` is the frozen A/B/C/D/E/F/H ablation suite
# (calibration / aggregation / retrieval / beam / subgroup / rescue /
# bootstrap / temperature), driven by src/cfla/final_ablation/fa_cpu_ablations.py.
# The other mechanism studies (candidate_conditioned/, factorization_generality/,
# cross_tokenizer_shared/, sports_confirmation/, multifactor_fusion/) are
# documented in docs/PIPELINE_STAGES.md and are invoked as their own scripts.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"

usage() { cat <<'USAGE'
Usage: bash scripts/run_ablations.sh --study final-ablation --cmd CMD
       bash scripts/run_ablations.sh --list

  --study S   final-ablation (the frozen CPU ablation suite)
  --cmd C     calibration | aggregation | retrieval | beam | subgroup |
              rescue | bootstrap | temperature
USAGE
}
STUDY=""; CMD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --study) STUDY="${2:?}"; shift 2 ;;
    --cmd) CMD="${2:?}"; shift 2 ;;
    --list) "$PY" - <<'PY'
print('studies: final-ablation')
print('  cmds: calibration aggregation retrieval beam subgroup rescue bootstrap temperature')
PY
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$STUDY" ] || { echo "error: --study is required" >&2; usage >&2; exit 2; }
[ -n "$CMD" ] || { echo "error: --cmd is required" >&2; usage >&2; exit 2; }
if [ ! -d "$ART" ]; then
  echo "FATAL: CFLA_ARTIFACT_ROOT does not exist: $ART" >&2
  echo "The ablation suite consumes the authors' frozen candidate-NLL / SID /" >&2
  echo "beam artifacts. Set CFLA_ARTIFACT_ROOT (docs/ENVIRONMENT.md)." >&2
  exit 2
fi
case "$STUDY" in
  final-ablation)
    "$PY" src/cfla/final_ablation/fa_cpu_ablations.py "$CMD" ;;
  *) echo "error: unknown --study $STUDY" >&2; exit 2 ;;
esac
echo "run_ablations: DONE ($STUDY/$CMD)"
