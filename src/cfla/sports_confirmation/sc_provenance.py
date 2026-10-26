#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round 4 close-out: run-level POST_RUN manifest + status.json, then the round-level
`_PROVENANCE.json` and `POST_RUN_ARTIFACT_MANIFEST.sha256`.

Ordering is load-bearing:
  1. run-level POST_RUN manifest  (hashes config/checkpoint/train log; written last
     among run-level files so the hash set is stable)
  2. run-level status.json
  3. round-level _PROVENANCE.json (enumerates every round-level output + its sha256)
  4. round-level POST_RUN_ARTIFACT_MANIFEST.sha256 (hashes everything, incl. provenance)

Writing the manifest before the file it hashes leaves a stale hash; that mistake was
made and caught in round 3, hence the explicit ordering here.

Schema follows outputs/disagreement_gate/_PROVENANCE.json so the two rounds are
directly comparable.
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
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# SC_OUT lets the smoke harness point the whole round-4 pipeline at a scratch tree,
# matching sc_pipeline.py / sc_gate_eval.py.
OUT = Path(os.environ.get('SC_OUT') or (REPO / 'outputs' / 'sports_confirmation'))
DEV = Path(os.environ.get('SC_DEV') or (REPO / 'outputs' / 'disagreement_gate'))
SPORTS = Path(_ART_ROOT + '/data/Sports_and_Outdoors')
PROV, MANIFEST = '_PROVENANCE.json', 'POST_RUN_ARTIFACT_MANIFEST.sha256'
PRE, POST = 'PRE_RUN_ARTIFACT_MANIFEST.sha256', 'POST_RUN_ARTIFACT_MANIFEST.sha256'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True,
                          cwd=str(REPO)).stdout.strip()


def rel(p):
    """Round-3 convention: round-relative path prefixed with 'outputs/'."""
    return 'outputs/' + str(Path(p).resolve().relative_to(OUT.resolve()))


def main():
    runs = sorted(p for p in (OUT / 'runs').glob('*') if p.is_dir())
    if len(runs) != 1:
        print(f'ERROR: expected exactly one run dir, found {len(runs)}: {runs}')
        return 1
    run = runs[0]
    rid = run.name
    tlog = load(run / 'train_log.json', {}) or {}
    epochs = tlog.get('epochs', []) or []

    # ---- 1. run-level POST_RUN manifest -------------------------------------
    run_files = [p for p in sorted(run.rglob('*'))
                 if p.is_file() and p.name not in (POST, 'status.json')]
    (run / POST).write_text('\n'.join(f'{sha(p)}  {p}' for p in run_files) + '\n')

    # ---- 2. run-level status.json -------------------------------------------
    beats = tlog.get('best_macro_valid_ndcg20')
    (run / 'status.json').write_text(json.dumps({
        'run_id': rid,
        'status': 'completed' if tlog else 'unknown',
        'detail': f'Sports shared T5 (spectral_k3 + rqvae3) trained with the audited '
                  f'development trainer; {len(epochs)} epochs, best epoch '
                  f'{tlog.get("best_epoch")} macro VALID NDCG@20 {beats}',
        'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }, indent=2) + '\n')

    # ---- 3. round-level _PROVENANCE.json ------------------------------------
    gman = load(DEV / 'GATE_FREEZE_MANIFEST.json', {}) or {}
    gman_sha = sha(DEV / 'GATE_FREEZE_MANIFEST.json')
    fman_p = OUT / 'SPORTS_CONFIRMATION_FREEZE_MANIFEST.json'
    fman = load(fman_p, {}) or {}
    scaler_p = OUT / 'valid' / 'GLOBAL_FEATURE_SCALER.json'
    scaler = load(scaler_p, {}) or {}

    # Sports SIDs: spectral_k3 was generated this round, rqvae3 is pre-existing
    sk_sid = OUT / 'sid' / 'spectral_k3' / 'sid.npy'
    rq_sid = SPORTS / 'Sports_and_Outdoors_t5_rqvae.npy'

    prov = {
        'study': 'pristine Sports_and_Outdoors confirmation of the frozen '
                 'candidate-conditioned factorization method',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git_sha': git('rev-parse', 'HEAD'),
        'branch': git('branch', '--show-current'),
        'host': platform.node(), 'platform': platform.platform(),

        'governance': {
            'sports_used_during_method_development': False,
            'gym_style_dev_dataset_available': False,
            'base_t5_trained_on_sports': True,
            'base_t5_of_development_datasets_modified': False,
            'sid_tokenizer_retrained_for_sports': False,
            'gate_retrained_on_sports': False,
            'gate_architecture_modified': False,
            'feature_schema_modified': False,
            'scoring_formula_modified': False,
            'hyperparameters_tuned_on_sports': False,
            'test_read_before_freeze': False,
            'sports_target_in_candidate_union': False,
            'target_in_gate_features': False,
            'paper_modified': False,
            'historical_outputs_overwritten': False,
            'development_candidate_nll_recomputed': False,
        },

        'sports_dataset': {
            'root': str(SPORTS),
            'users_per_split': 35598, 'items': 18357,
            'spectral_k3': {
                'path': str(sk_sid), 'sha256': sha(sk_sid),
                'shape': list(np.load(sk_sid).shape),
                'generated_this_round': True,
                'generator': 'analysis/mechanism/sports_confirmation/sc_generate_sid.py',
                'certified_by': 'reproduced Beauty K=3 byte-for-byte before use',
            },
            'rqvae3': {'path': str(rq_sid), 'sha256': sha(rq_sid),
                       'generated_this_round': False},
            'stale_artifact_not_used': {
                'path': str(SPORTS / 'Sports_and_Outdoors_t5_spectral_v2.npy'),
                'sha256': sha(SPORTS / 'Sports_and_Outdoors_t5_spectral_v2.npy'),
                'reason': 'K=2 granularity, wrong dimensionality for the F1 control-token '
                          'layout; audited in SPORTS_SID_AUDIT.md and excluded',
            },
        },

        'frozen_shared_model': {
            'run_directory': f'outputs/{run.name}',
            'checkpoint': f'outputs/{run.name}/shared_model.pt',
            'checkpoint_sha256': sha(run / 'shared_model.pt'),
            'trainer': 'analysis/mechanism/cross_tokenizer/xt_train_shared.py called '
                       'unchanged; only scoped_sid_path was shadowed for the Sports '
                       'spectral_k3 array',
            'epochs_run': len(epochs),
            'best_epoch': tlog.get('best_epoch'),
            'best_macro_valid_ndcg20': beats,
            'vocab_size': tlog.get('config', {}).get('vocab_size'),
            'control_tokens': {'spectral_k3': 1025, 'rqvae3': 1026},
        },

        'frozen_gate': {
            'name': gman.get('gate', {}).get('name'),
            'run_directory': gman.get('gate', {}).get('run_directory'),
            'checkpoint': gman.get('gate', {}).get('checkpoint'),
            'checkpoint_sha256': gman.get('gate', {}).get('checkpoint_sha256'),
            'params': gman.get('gate', {}).get('params'),
            'architecture': gman.get('gate', {}).get('architecture'),
            'lambda_g': gman.get('gate', {}).get('lambda_g'),
            'training_seed': gman.get('gate', {}).get('training_seed'),
            'freeze_manifest': 'outputs/disagreement_gate/GATE_FREEZE_MANIFEST.json',
            'freeze_manifest_sha256': gman_sha,
            'retrained_this_round': False,
        },

        'features': {
            'count': gman.get('features', {}).get('dimension'),
            'list': gman.get('features', {}).get('list'),
            'uses_target': False,
            'development_scaler': 'outputs/disagreement_gate/features/FEATURE_SCALER.json',
            'sports_scaler': str(scaler_p),
            'sports_scaler_sha256': sha(scaler_p),
            'sports_scaler_protocol': scaler.get('protocol'),
            'sports_scaler_fitted_on': scaler.get('fitted_on'),
            'sports_scaler_rows': scaler.get('total_rows'),
            'sports_contributed_to_scaler': scaler.get('sports_used'),
            'unseen_domain_onehot': [0, 0, 0, 0],
            'dataset_onehot_dimension_added': False,
        },

        'candidate_nll': {
            'sports_valid': f'outputs/{OUT.name}/valid/candidate_nll.parquet',
            'sports_test': f'outputs/{OUT.name}/test/candidate_nll.parquet',
            'development_artifacts_reused': (
                'outputs/candidate_conditioned/runs/'
                'multi_candidate_nll_seed2025_20260914T194657'),
            'development_artifacts_recomputed': False,
            'sports_computed_this_round': True,
            'sports_rationale': 'Sports had no candidate artefacts before this round; '
                                'they are produced from the Sports shared model. The '
                                'four development datasets were reused read-only.',
        },

        'protocol': {
            'design': 'zero-shot domain transfer: the method (score formulas, gate '
                      'weights, feature schema) is frozen from development and applied '
                      'unchanged to Sports',
            'sports_valid_uses': ['candidate NLL scaler (mu/sigma per factorization)',
                                  'shared-model checkpoint selection'],
            'sports_test_uses': ['single confirmatory evaluation after freeze'],
            'pre_registered_claim_policy': 'harness section 33',
            'freeze_manifest': f'outputs/{OUT.name}/SPORTS_CONFIRMATION_FREEZE_MANIFEST.json',
            'freeze_manifest_sha256': sha(fman_p) if fman_p.exists() else None,
            'test_read': fman.get('sports_test_read'),
            'valid_nll_scaler': fman.get('valid_nll_scaler'),
            'bootstrap': 'paired user-level bootstrap, 5000 resamples, seed 20260912, 95% CI',
        },

        'run_level_manifests': {'pre': PRE, 'post': POST, 'directory': f'outputs/{run.name}'},
        'outputs': {},
    }

    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in (PROV, MANIFEST):
            prov['outputs'][rel(p)] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    (OUT / PROV).write_text(json.dumps(prov, indent=2, default=str) + '\n')

    # ---- 4. round-level POST_RUN manifest (hashes provenance too) -----------
    shots = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != MANIFEST]
    (OUT / MANIFEST).write_text('\n'.join(f'{sha(p)}  {p}' for p in shots) + '\n')

    print(f'run  POST_RUN: {len(run_files)} files')
    print(f'round PROVENANCE: {len(prov["outputs"])} outputs')
    print(f'round POST_RUN: {len(shots)} files')
    print(f'checkpoint sha256: {prov["frozen_shared_model"]["checkpoint_sha256"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
