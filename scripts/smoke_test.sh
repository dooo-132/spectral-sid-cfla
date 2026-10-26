#!/usr/bin/env bash
# CPU-only smoke test of the release.  No dataset, no checkpoint, no GPU.
#
#   bash scripts/smoke_test.sh
#
# Runs, in order:
#   1. environment check                  (versions + path roots)
#   2. byte-compile every python file     (syntax)
#   3. unit tests                         (tests/run_tests.py)
#   4. synthetic end-to-end demo          (examples/synthetic_demo.py)
#
# Everything runs on synthetic inputs; nothing is written into the repository
# (the tests and the demo use throw-away temporary roots).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"

echo "=================================================================="
echo " CFLA release smoke test   $(date -Is)"
echo " repo: $REPO"
echo "=================================================================="

echo
echo "--- 1/4 environment ---"
bash scripts/check_environment.sh

echo
echo "--- 2/4 byte-compile ---"
# compile in memory (writing .pyc into the tree would pollute a fresh checkout)
"$PY" - <<'PY'
import pathlib, sys
bad = []
files = [p for p in pathlib.Path('.').rglob('*.py')
         if '.git' not in p.parts and '__pycache__' not in p.parts]
for p in files:
    try:
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    except SyntaxError as e:
        bad.append((str(p), str(e)))
print(f"compiled {len(files)} python files in memory, failures: {len(bad)}")
for b in bad:
    print("FAIL", b)
sys.exit(1 if bad else 0)
PY

echo
echo "--- 3/4 unit tests (synthetic) ---"
"$PY" tests/run_tests.py

echo
echo "--- 4/4 synthetic end-to-end demo ---"
"$PY" examples/synthetic_demo.py

echo
echo "=================================================================="
echo " SMOKE TEST PASSED"
echo "=================================================================="
