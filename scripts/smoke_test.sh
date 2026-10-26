#!/usr/bin/env bash
# CPU-only smoke test of the release.  No dataset, no checkpoint, no GPU.
#
#   bash scripts/smoke_test.sh
#
# Runs: environment check -> in-memory byte-compile of every python file ->
# import smoke test of the core packages.  (The unit-test suite and the
# synthetic demo are intentionally not shipped in this release.)
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
echo "--- 1/3 environment ---"
bash scripts/check_environment.sh

echo
echo "--- 2/3 byte-compile (in memory, nothing written) ---"
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
echo "--- 3/3 import smoke test ---"
"$PY" - <<'PY'
import warnings, importlib
warnings.filterwarnings('ignore')
mods = ['src', 'src.sid', 'src.sid.generator', 'src.tiger', 'src.tiger.train',
        'src.tiger.dump_beams', 'src.fusion.common', 'src.analysis.teacher_forced',
        'src.utils.repo_paths', 'src.runner', 'src.cfla', 'src.cfla.cfla']
ok = fail = 0
for m in mods:
    try:
        importlib.import_module(m); ok += 1
    except Exception as e:
        print('FAIL', m, type(e).__name__, str(e)[:120]); fail += 1
print(f"imports: {ok} ok, {fail} failed")
raise SystemExit(1 if fail else 0)
PY

echo
echo "=================================================================="
echo " SMOKE TEST PASSED"
echo "=================================================================="
