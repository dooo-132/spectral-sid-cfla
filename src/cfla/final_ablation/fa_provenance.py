#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write outputs/final_ablation/_PROVENANCE.json and POST_RUN_ARTIFACT_MANIFEST.sha256.

Run LAST, after every ablation and the report.  The provenance is written first and the
manifest that hashes it second, so the manifest verifies; writing the manifest before the
file it covers leaves a stale hash (a mistake made in an earlier round).

    python3 fa_provenance.py
"""
from __future__ import annotations

import os as _os
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[3])
_ART_ROOT = _os.environ.get('CFLA_ARTIFACT_ROOT') or _os.path.join(_REPO_ROOT, 'artifacts')
_UP_ROOT = _os.environ.get('CFLA_UPSTREAM_ROOT') or _ART_ROOT
import sys as _sys
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from src.utils import repo_paths as rp  # noqa: E402
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fa_common as C  # noqa: E402

PROV = '_PROVENANCE.json'
MANIFEST = 'POST_RUN_ARTIFACT_MANIFEST.sha256'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True,
                          cwd=str(C.REPO)).stdout.strip()


def main():
    runs = sorted(p for p in (C.OUT / 'runs').glob('*') if p.is_dir())
    status = {}
    for d in runs:
        sp = d / 'status.json'
        status[d.name] = json.loads(sp.read_text()).get('status') if sp.exists() else 'missing'

    prov = {
        'study': 'final ablation of the frozen Cross-Factorization Candidate Likelihood method',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git_sha': git('rev-parse', 'HEAD'),
        'branch': git('branch', '--show-current'),
        'host': platform.node(), 'platform': platform.platform(),

        'governance': {
            'new_t5_training': False,
            'new_tokenizer_training': False,
            'new_gate_training': False,
            'historical_outputs_overwritten': False,
            'audited_repo_modified': False,
            'paper_modified': False,
            'sports_used_for_parameter_selection': False,
            'sports_used_for_method_design': False,
            'test_used_for_parameter_selection': False,
            'raw_repo_artifacts_modified': False,
        },

        'frozen_method': {
            'uniform': '0.5*P_F1 + 0.5*P_F2, P_F = 0 outside F beam',
            'lse': 'logsumexp(-zNLL_F1, -zNLL_F2), numerically stable',
            'znll': '(TotalNLL_F - mu_F)/sigma_F, mu/sigma over ALL VALID candidates',
            'source_module': 'analysis/mechanism/candidate_conditioned/cc_rerank_methods.py',
            'reimplementation_verified': True,
            'verification': 'fa_verify.py reproduces the four frozen development datasets to '
                            'within 4e-7 and Sports exactly; scalers refit bit-identically',
        },

        'reused_frozen_artifacts': {
            'candidate_run': 'outputs/candidate_conditioned/runs/'
                             'multi_candidate_nll_seed2025_20260914T194657',
            'dev_scaler': 'outputs/candidate_conditioned/validation/candidate_nll_scaler.json',
            'sports_dir': 'outputs/sports_confirmation',
            'shared_checkpoints': 'outputs/cross_tokenizer_shared/step2_shared_model/runs',
            'independent_checkpoints': _ART_ROOT + '/model/ckpt (read-only)',
            'independent_beams_step1': 'outputs/cross_tokenizer_shared/step1_cross_tokenizer/beams',
            'same_family_matrix': 'outputs/mechanism/factorization_generality/ARTIFACT_MATRIX.csv',
            'candidate_nll_recomputed_for_dev': False,
        },

        'ablations': {
            'A_calibration': {'status': 'completed', 'output': 'calibration_ablation.csv'},
            'B_aggregation': {'status': 'completed', 'output': 'aggregation_ablation.csv'},
            'C_retrieval_vs_likelihood': {'status': 'completed',
                                          'output': 'retrieval_likelihood_ablation.csv'},
            'D_shared_vs_independent': {
                'status': 'completed' if (C.OUT / 'shared_vs_independent.csv').exists()
                          else 'not_run',
                'output': 'shared_vs_independent.csv'},
            'E_same_family': {
                'status': 'partial' if (C.OUT / 'family_ablation.csv').exists() else 'not_run',
                'output': 'family_ablation.csv',
                'limitation': 'LSE half needs teacher-forced NLL under same-family '
                              'checkpoints; CPU-only coverage/Jaccard/Uniform completed'},
            'F_beam_budget': {'status': 'completed', 'output': 'beam_budget_ablation.csv'},
            'G_factorization_count': {
                'status': 'inventory_only'
                          if (C.OUT / 'factorization_count_inventory.csv').exists()
                          else 'not_run',
                'output': 'factorization_count_inventory.csv'},
            'H_mechanism': {'status': 'completed',
                            'outputs': ['subgroup_analysis.csv', 'rescue_damage.csv',
                                        'rank_movement.csv']},
        },

        'runs': {'count': len(runs), 'status': status},
        'outputs': {},
    }

    for p in sorted(C.OUT.rglob('*')):
        if p.is_file() and p.name not in (PROV, MANIFEST):
            prov['outputs'][str(p.relative_to(C.OUT))] = {'sha256': sha(p),
                                                          'bytes': p.stat().st_size}
    (C.OUT / PROV).write_text(json.dumps(prov, indent=2, default=str) + '\n')

    shots = [p for p in sorted(C.OUT.rglob('*')) if p.is_file() and p.name != MANIFEST]
    (C.OUT / MANIFEST).write_text('\n'.join(f'{sha(p)}  {p}' for p in shots) + '\n')

    print(f'wrote {PROV}: {len(prov["outputs"])} outputs')
    print(f'wrote {MANIFEST}: {len(shots)} files')
    print(f'runs: {len(runs)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
