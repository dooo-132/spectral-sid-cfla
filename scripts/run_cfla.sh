#!/usr/bin/env bash
# Stage 6 -- CFLA (Cross-Factorization Likelihood Aggregation) and the paper's
# M5 pipeline, by stage.  These are CPU-only consumers of frozen artifacts
# (beam dumps, SID arrays, checkpoints, candidate-NLL parquet).
#
#   bash scripts/run_cfla.sh --stage m5 --check
#   bash scripts/run_cfla.sh --stage candidate_nll --datasets Beauty --device cuda
#   bash scripts/run_cfla.sh --stage m5_cfla
#   bash scripts/run_cfla.sh --stage table1
#
# Stages and their real entry points:
#   candidate_nll  teacher-forced candidate NLL for the 5 factorizations
#                  -> src/cfla/paper_pipeline/score_m5_candidate_nll.py
#   m5             candidate union + per-factorization strategies + bootstrap
#                  for M5 Uniform / structure-aware / confidence / grid
#                  -> src/cfla/paper_pipeline/build_m5_rq5.py
#   m5_cfla        zNLL calibration + LSE aggregation rows of M5 (wraps the
#                  frozen driver without editing it)
#                  -> src/cfla/paper_pipeline/build_m5_cfla.py
#   efficiency     measured inference complexity / latency
#                  -> src/cfla/paper_pipeline/measure_inference_complexity.py
#   codebook       codebook-size sensitivity
#                  -> src/cfla/paper_pipeline/build_codebook_sensitivity.py
#   table1         final Table 1 from frozen CSVs
#                  -> src/cfla/paper_results/build_table1_final.py
#   efficiency_table + codebook_figure
#                  -> src/cfla/paper_results/{make_table_efficiency,make_fig_codebook}.py
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
ART="${CFLA_ARTIFACT_ROOT:-$REPO/artifacts}"

usage() { cat <<'USAGE'
Usage: bash scripts/run_cfla.sh --stage STAGE [stage options]

Stages: candidate_nll | m5 | m5_cfla | efficiency | codebook |
        table1 | efficiency_table | codebook_figure

Shared environment (see docs/ENVIRONMENT.md):
  CFLA_ARTIFACT_ROOT  authors' frozen artifact root (required by all stages
                      except the table/figure builders, which read CFLA_OUTPUT_ROOT)
  CFLA_OUTPUT_ROOT    writable output root (default <repo>/outputs)
  CFLA_PYTHON         python interpreter (default python3)

All remaining arguments are passed verbatim to the stage's real entry point,
e.g. `--datasets Beauty --dry-run --check`.
USAGE
}
STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done
[ -n "$STAGE" ] || { echo "error: --stage is required" >&2; usage >&2; exit 2; }
case "$STAGE" in
  candidate_nll) ENTRY=src/cfla/paper_pipeline/score_m5_candidate_nll.py ;;
  m5) ENTRY=src/cfla/paper_pipeline/build_m5_rq5.py ;;
  m5_cfla) ENTRY=src/cfla/paper_pipeline/build_m5_cfla.py ;;
  efficiency) ENTRY=src/cfla/paper_pipeline/measure_inference_complexity.py ;;
  codebook) ENTRY=src/cfla/paper_pipeline/build_codebook_sensitivity.py ;;
  table1) ENTRY=src/cfla/paper_results/build_table1_final.py ;;
  efficiency_table) ENTRY=src/cfla/paper_results/make_table_efficiency.py ;;
  codebook_figure) ENTRY=src/cfla/paper_results/make_fig_codebook.py ;;
  *) echo "error: unknown --stage $STAGE" >&2; usage >&2; exit 2 ;;
esac
HAS_CHECK=0
for _a in "$@"; do [ "$_a" = "--check" ] && HAS_CHECK=1; done
case "$STAGE" in
  candidate_nll|m5|m5_cfla|efficiency|codebook)
    if [ ! -d "$ART" ] && [ "$HAS_CHECK" = 0 ]; then
      echo "FATAL: CFLA_ARTIFACT_ROOT does not exist: $ART" >&2
      echo "This stage consumes the authors' frozen artifacts (beam dumps, SID" >&2
      echo "arrays, checkpoints, candidate-NLL parquet); see docs/DATA_FORMAT.md." >&2
      echo "Set CFLA_ARTIFACT_ROOT, or run the stage with --check to list what is missing." >&2
      exit 2
    fi ;;
esac
echo "run_cfla stage=$STAGE -> $ENTRY"
"$PY" "$ENTRY" "$@"
echo "run_cfla: DONE ($STAGE)"
