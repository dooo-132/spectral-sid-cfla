#!/usr/bin/env bash
# Print the interpreter, the imported package versions and the resolved path
# roots, then fail if a required runtime dependency is missing.
#
#   bash scripts/check_environment.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PY="${CFLA_PYTHON:-python3}"

"$PY" - <<'PY'
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.utils import repo_paths as rp

print('python           ', platform.python_version(), '|', platform.platform())
print('REPO_ROOT        ', rp.REPO_ROOT)
print('CFLA_ARTIFACT_ROOT', rp.ARTIFACT_ROOT, '(exists:', rp.ARTIFACT_ROOT.exists(), ')')
print('CFLA_OUTPUT_ROOT  ', rp.OUTPUT_ROOT)
print()
required = ['torch', 'transformers', 'numpy', 'scipy', 'sklearn', 'pandas',
            'yaml', 'tqdm']
optional = ['faiss', 'matplotlib', 'sentence_transformers', 'pyarrow']
ok = True
for mod in required + optional:
    tag = 'required' if mod in required else 'optional'
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        print(f'  [{tag}] {mod:22s} {ver}')
    except Exception as e:  # noqa: BLE001
        print(f'  [{tag}] {mod:22s} MISSING ({type(e).__name__})')
        if tag == 'required':
            ok = False
print()
print('core imports:')
for mod in ('src.sid.generator', 'src.tiger.train', 'src.tiger.dataset',
            'src.fusion.common', 'src.analysis.teacher_forced',
            'src.utils.experiment_config', 'src.runner'):
    importlib.import_module(mod)
    print(f'  ok  {mod}')
print()
print('ENVIRONMENT_OK' if ok else 'ENVIRONMENT_MISSING_REQUIRED_PACKAGES')
raise SystemExit(0 if ok else 1)
PY
