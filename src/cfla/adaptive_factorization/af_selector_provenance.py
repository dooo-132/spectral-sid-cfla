#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble `selector/provenance.json`: every input the target-free selector consumed,
with its SHA-256, plus the frozen selection decisions.

Read-only. Run after `af_eval_test.py`:
  python3 af_selector_provenance.py --selectordir outputs/adaptive_factorization/selector
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_router_train import DATASET_ORDER  # noqa: E402
from xt_common import STEP2, sha256_file, sid_path  # noqa: E402

FACTORIZATIONS = ('spectral_k3', 'rqvae3')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selectordir', required=True)
    a = ap.parse_args(argv)
    sel = Path(a.selectordir)

    def git_sha():
        try:
            return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, cwd=str(HERE.parents[2])).stdout.strip()
        except Exception:
            return None

    dev = json.loads((sel / 'router_dev_summary.json').read_text())
    freeze_p = sel / 'SELECTOR_FREEZE_MANIFEST.json'
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else None
    nllman_p = sel / 'features' / 'nll' / 'NLL_MANIFEST.json'
    nllman = json.loads(nllman_p.read_text()) if nllman_p.exists() else None
    featman_p = sel / 'features' / 'FEATURE_BUILD_PROVENANCE.json'
    featman = json.loads(featman_p.read_text()) if featman_p.exists() else None

    prov = {
        'study': 'target-free adaptive factorization selector',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git_sha': git_sha(),
        'host': platform.node(), 'platform': platform.platform(),
        'governance': {
            'target_in_selector_features': False,
            'user_id_in_selector_features': False,
            'target_nll_in_selector_features': False,
            'test_metrics_read_before_freeze': False,
            'frozen_shared_checkpoints_retrained': False,
            'historical_artifacts_modified': False,
        },
        'frozen_shared_checkpoints': {},
        'frozen_test_beam_dumps': {},
        'sid_arrays': {},
        'generated_validation_beams_and_history': {},
        'teacher_forced_nll': {},
        'selector': {
            'chosen_router': freeze.get('chosen_router') if freeze else None,
            'chosen_router_sha256': freeze.get('chosen_router_sha256') if freeze else None,
            'scaler_hash': dev.get('scaler_hash'),
            'feature_names_h': dev.get('feature_names_h'),
            'feature_names_p': dev.get('feature_names_p'),
            'router_dev_results': dev.get('results'),
            'router_dev_significance_vs_uniform':
                dev.get('router_dev_significance_vs_uniform'),
            'static_weight': freeze.get('static_weight') if freeze else None,
        },
        'user_split': {
            'file': 'router_user_split.csv',
            'sha256': sha256_file(sel / 'router_user_split.csv'),
            'seed': dev.get('split_seed'), 'fractions': dev.get('split_fractions'),
            'router_train_n': dev.get('router_train_n'),
            'router_dev_n': dev.get('router_dev_n'),
            'router_holdout_n': dev.get('router_holdout_n'),
        },
        'outputs': {},
    }

    for ds in DATASET_ORDER:
        ck = STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'
        prov['frozen_shared_checkpoints'][ds] = {
            'path': str(ck), 'sha256': sha256_file(ck)}
        for f in FACTORIZATIONS:
            src = STEP2 / 'beams' / f'{ds}_{f}_test.npz'
            dst = sel / 'features' / 'beams' / f'{ds}_{f}_test.npz'
            prov['frozen_test_beam_dumps'][f'{ds}|{f}'] = {
                'frozen_original': str(src), 'frozen_original_sha256': sha256_file(src),
                'selector_copy': str(dst), 'selector_copy_sha256': sha256_file(dst),
                'byte_identical': sha256_file(src) == sha256_file(dst)}
            prov['sid_arrays'][f'{ds}|{f}'] = {
                'path': str(sid_path(f, ds)), 'sha256': sha256_file(sid_path(f, ds))}
            prov['generated_validation_beams_and_history'][f'{ds}|{f}'] = {
                'beams': str(sel / 'features' / 'beams' / f'{ds}_{f}_valid.npz'),
                'beams_sha256': sha256_file(
                    sel / 'features' / 'beams' / f'{ds}_{f}_valid.npz'),
                'history_repr': str(sel / 'features' / 'history_repr' / f'{ds}_{f}_valid.npz'),
                'history_repr_sha256': sha256_file(
                    sel / 'features' / 'history_repr' / f'{ds}_{f}_valid.npz'),
            }
        for split in ('valid', 'test'):
            p = sel / 'features' / 'nll' / f'{ds}_{FACTORIZATIONS[0]}_{split}.npz'
            if p.exists():
                prov['teacher_forced_nll'][f'{ds}|{split}'] = {
                    'spectral_k3': sha256_file(p),
                    'rqvae3': sha256_file(
                        sel / 'features' / 'nll' / f'{ds}_{FACTORIZATIONS[1]}_{split}.npz'),
                    'used_as_feature': False,
                }
        for split in ('valid', 'test'):
            p = sel / 'features' / f'{ds}_{split}.npz'
            if p.exists():
                prov['outputs'][f'{ds}|{split}'] = {'path': str(p),
                                                    'sha256': sha256_file(p)}

    prov['feature_build_manifest'] = featman
    prov['nll_manifest'] = nllman
    prov['freeze_manifest_file'] = {
        'path': str(freeze_p), 'sha256': sha256_file(freeze_p)} if freeze_p.exists() else None
    for name in ('test_results.csv', 'bootstrap.csv', 'preference_diagnostics.csv',
                 'nll_diagnostics.csv', 'subgroup_results.csv', 'robustness_results.csv',
                 'efficiency.csv', 'router_dev_results.csv'):
        p = sel / name
        if p.exists():
            prov.setdefault('result_files', {})[name] = {
                'path': str(p), 'sha256': sha256_file(p)}

    out = sel / 'provenance.json'
    out.write_text(json.dumps(prov, indent=2, default=str))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
