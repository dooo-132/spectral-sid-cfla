#!/usr/bin/env bash
# End-to-end CPU smoke test on SYNTHETIC data (no real dataset, no GPU).
#
#   bash scripts/smoke_pipeline.sh [--epochs N] [--keep]
#
# It runs the complete shipped pipeline on a tiny synthetic corpus:
#   synthetic raw reviews -> splits + embeddings -> spectral SIDs (K=2,3)
#   -> two tiny T5 models (CPU training, few epochs) -> beam dumps
#   -> CFLA (candidate union, teacher-forced NLL, validation calibration,
#      log-sum-exp aggregation, Recall/NDCG evaluation)
# and asserts that every stage produces its artifacts and that the final metrics
# are finite.  Everything is written to a temporary directory; nothing in the
# repository is modified.  Do not interpret the resulting numbers as results --
# they come from a synthetic corpus and a 1-minute CPU training run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"
EPOCHS=3
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) echo "Usage: bash scripts/smoke_pipeline.sh [--epochs N] [--keep]"; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
done

TMP="$(mktemp -d /tmp/cfla_smoke_XXXXXX)"
export CFLA_ARTIFACT_ROOT="$TMP/artifacts"
export CFLA_OUTPUT_ROOT="$TMP/outputs"
export CFLA_RAW_DATA="$TMP/raw"
export CFLA_PYTHON="$PY"
mkdir -p "$CFLA_RAW_DATA" "$CFLA_OUTPUT_ROOT"
echo "synthetic smoke workspace: $TMP"

step() { echo; echo "=== $* ==="; }

step "1/6 data: synthetic raw reviews -> splits + item embeddings"
bash scripts/prepare_data.sh --dataset Beauty --stage all --synthetic-smoke

step "2/6 SID construction: spectral K=2 and K=3 (synthetic embeddings)"
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 2
bash scripts/build_sid.sh --tokenizer spectral --dataset Beauty --k 3

step "3/6 tiny CPU training: one T5 per factorization (${EPOCHS} epochs, synthetic)"
for K in 2 3; do
  L=$((K + 1))
  bash scripts/train.sh --model t5_single --dataset Beauty --k "$K" \
      --code-path "$CFLA_OUTPUT_ROOT/sids/Beauty/nonneg_pearson/k${K}/seed2025/sid.npy" \
      --device cpu --epochs "$EPOCHS" --batch-size 128 --lr 3e-3 \
      --beam-size 8 --seed 2025 --num-workers 0
  test -f "$CFLA_OUTPUT_ROOT/runs/t5_Beauty_k${K}_sid${L}_seed2025.pth" || {
      echo "FATAL: training produced no checkpoint for K=$K" >&2; exit 1; }
done

step "4/6 beam candidates: valid + test dumps for both factorizations (CPU)"
for K in 2 3; do
  L=$((K + 1))
  for SPLIT in valid test; do
    bash scripts/generate_candidates.sh --dataset Beauty --k "$K" --split "$SPLIT" \
        --code-path "$CFLA_OUTPUT_ROOT/sids/Beauty/nonneg_pearson/k${K}/seed2025/sid.npy" \
        --ckpt-path "$CFLA_OUTPUT_ROOT/runs/t5_Beauty_k${K}_sid${L}_seed2025.pth" \
        --output-path "$CFLA_OUTPUT_ROOT/beam_dumps/Beauty/${SPLIT}_k${K}_sid${L}.npz" \
        --beam-size 12 --infer-size 48 --device cpu --seed 2025 --num-workers 0 --allow-low-decode
  done
done

step "5/6 CFLA pool config"
CFG="$TMP/cfla_pool_smoke.yaml"
cat > "$CFG" <<YAML
dataset: Beauty
dataset_dir: ${CFLA_ARTIFACT_ROOT}/data/Beauty
codebook_size: 256
max_len: 20
output_dir: ${CFLA_OUTPUT_ROOT}/cfla
factorizations:
  - name: spectral_k2
    sid: ${CFLA_OUTPUT_ROOT}/sids/Beauty/nonneg_pearson/k2/seed2025/sid.npy
    ckpt: ${CFLA_OUTPUT_ROOT}/runs/t5_Beauty_k2_sid3_seed2025.pth
    beam_valid: ${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/valid_k2_sid3.npz
    beam_test: ${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/test_k2_sid3.npz
  - name: spectral_k3
    sid: ${CFLA_OUTPUT_ROOT}/sids/Beauty/nonneg_pearson/k3/seed2025/sid.npy
    ckpt: ${CFLA_OUTPUT_ROOT}/runs/t5_Beauty_k3_sid4_seed2025.pth
    beam_valid: ${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/valid_k3_sid4.npz
    beam_test: ${CFLA_OUTPUT_ROOT}/beam_dumps/Beauty/test_k3_sid4.npz
YAML

step "6/6 CFLA: union -> candidate NLL -> calibration -> LSE -> evaluation"
bash scripts/run_cfla.sh --config "$CFG" --device cpu --batch-size 64 \
    --bootstrap 200 --overwrite

# ---- assertions -----------------------------------------------------------
test -f "$CFLA_OUTPUT_ROOT/cfla/metrics.json"
test -f "$CFLA_OUTPUT_ROOT/cfla/per_user.csv"
"$PY" - "$CFLA_OUTPUT_ROOT/cfla/metrics.json" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
test = d["splits"].get("test", {})
assert test, "no test split in metrics.json"
for method, metrics in test["metrics"].items():
    for k, v in metrics.items():
        assert math.isfinite(v), f"non-finite metric {method}/{k}={v}"
        assert 0.0 <= v <= 1.0, f"metric out of range {method}/{k}={v}"
    print(f"  {method:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
cov = test.get("union_coverage")
print(f"  union coverage (synthetic, tiny models): {cov:.3f}")
print("SMOKE PIPELINE: metrics are finite and in range")
PY

if [ "$KEEP" = 1 ]; then
  echo "artifacts kept in: $TMP"
else
  rm -rf "$TMP"
fi
echo
echo "=================================================================="
echo " SMOKE PIPELINE PASSED (synthetic data, CPU, ${EPOCHS} training epochs)"
echo "=================================================================="
