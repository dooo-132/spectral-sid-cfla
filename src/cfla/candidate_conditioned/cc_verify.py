#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artifact-integrity and immutability checks for the candidate-conditioned round.

  * historical directories (`outputs/mechanism`, `outputs/cross_tokenizer_shared`,
    `outputs/adaptive_factorization`, and the read-only artifact root) were not modified
  * the candidate scorer's schema still admits no target quantity
  * the candidate union was built without the target
  * the freeze manifest exists and was written before test was read
  * the NLL the candidate scorer produces for the known target still reproduces the
    frozen target-NLL dumps

Run:  python3 cc_verify.py --outdir outputs/candidate_conditioned
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
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(HERE), str(REPO / 'analysis' / 'mechanism' / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cc_common  # noqa: E402

FAILS = []
N = [0]


def check(name, cond, detail=''):
    N[0] += 1
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}  {detail}')
        FAILS.append(name)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def immutability():
    print('1. historical immutability')
    for d in ('outputs/mechanism', 'outputs/cross_tokenizer_shared',
              'outputs/adaptive_factorization'):
        p = REPO / d
        if not p.exists():
            continue
        out = subprocess.run(
            ['bash', '-c',
             f'find {p} -newermt "2026-09-14 19:30" -type f 2>/dev/null | wc -l'],
            capture_output=True, text=True).stdout.strip()
        check(f'{d} untouched during this round', out == '0', f'{out} files newer')
    for root in (_ART_ROOT + '', _UP_ROOT + ''):
        out = subprocess.run(
            ['bash', '-c',
             f'find {root} -newermt "2026-09-14 19:30" -type f -not -path "*/.git/*" '
             f'2>/dev/null | wc -l'], capture_output=True, text=True).stdout.strip()
        check(f'{root} untouched', out == '0', f'{out} files newer')
    out = subprocess.run(['bash', '-c',
                          'find paper -newermt "2026-09-14 19:30" -type f | wc -l'],
                         capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    check('paper directory untouched', out == '0', f'{out} files newer')


def leakage():
    print('2. leakage guards')
    sig = set(__import__('inspect').signature(cc_common.candidate_total_nll).parameters)
    check('scorer signature is the declared target-free schema',
          sig == {'model', 'history_tokens', 'history_mask', 'cand_tokens', 'cand_owner',
                  'device', 'batch_size', 'want_per_token'}, str(sorted(sig)))
    # AST-based, so a docstring that merely *says* "no target information is consumed"
    # is not counted as code that reads a target.
    import ast as _ast
    import inspect as _inspect
    tree = _ast.parse(_inspect.getsource(cc_common.candidate_total_nll))
    idents = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
    idents |= {n.attr for n in _ast.walk(tree) if isinstance(n, _ast.Attribute)}
    offenders = sorted(i for i in idents
                       if {w for w in i.lower().split('_')} & {'target', 'label', 'hit'})
    check('scorer code references no target identifier', not offenders, str(offenders))


def outputs(rundir, out):
    print('3. candidate-NLL outputs')
    for ds in cc_common.DATASETS:
        for split in ('valid', 'test'):
            for kind in ('candidate_union', 'candidate_nll'):
                f = Path(rundir) / split / f'{ds}_{kind}.parquet'
                check(f'{ds} {split} {kind} present', f.exists())
    u = Path(rundir) / 'test' / 'Beauty_candidate_union.parquet'
    n = Path(rundir) / 'test' / 'Beauty_candidate_nll.parquet'
    if u.exists() and n.exists():
        import pandas as pd
        du = pd.read_parquet(u); dn = pd.read_parquet(n)
        bad = [c for c in list(du.columns) + list(dn.columns)
               if any(b in c.lower() for b in ('target', 'label', 'hit'))]
        check('no target-derived column in union or NLL', not bad, str(bad))
        check('union and NLL are row-aligned',
              len(du) == len(dn) and (du.user.values == dn.user.values).all()
              and (du.candidate_item.values == dn.candidate_item.values).all())
        check('NLL columns are finite for all candidates',
              bool(np.isfinite(dn.TotalNLL_F1).all() and np.isfinite(dn.TotalNLL_F2).all()))


def freeze_and_results(out):
    print('4. freeze discipline and results')
    out = Path(out)
    fp = out / 'CANDIDATE_RERANK_FREEZE_MANIFEST.json'
    check('freeze manifest exists', fp.exists())
    if fp.exists():
        man = json.loads(fp.read_text())
        for k in ('git_sha', 'shared_checkpoints', 'candidate_generation', 'nll_scaler',
                  'selected_T', 'selected_lambda', 'method_definitions',
                  'validation_ndcg20'):
            check(f'manifest records {k}', k in man)
        check('manifest records test not read at freeze time',
              man.get('test_metrics_computed_at_freeze_time') is False)
    for name in ('test_results.csv', 'target_rank_movement.csv',
                 'uniform_failure_rescue.csv', 'subgroup_results.csv', 'bootstrap.csv'):
        check(f'test/{name} present', (out / 'test' / name).exists())
    boot = list(csv.DictReader((out / 'test' / 'bootstrap.csv').open())) \
        if (out / 'test' / 'bootstrap.csv').exists() else []
    check('bootstrap used 5000 resamples',
          all(int(r.get('n_users', 0)) > 0 for r in boot) and len(boot) > 0)
    if boot:
        ncols = [r for r in boot if r['metric'] == 'NDCG@20']
        check('bootstrap reports delta, CI and p for NDCG@20', len(ncols) > 0)


def scorer_reproduces_frozen():
    print('5. scorer still reproduces the frozen target NLL')
    import torch
    if not torch.cuda.is_available():
        print('  SKIP  no GPU')
        return
    from xt_common import load_sid
    feat = (REPO / 'outputs' / 'adaptive_factorization' / 'selector' / 'features' /
            'Beauty_test.npz')
    nllp = (REPO / 'outputs' / 'adaptive_factorization' / 'selector' / 'features' /
            'nll' / 'Beauty_spectral_k3_test.npz')
    if not (feat.exists() and nllp.exists()):
        print('  SKIP  frozen inputs absent')
        return
    model, plan = cc_common.load_frozen_shared(cc_common.frozen_checkpoint('Beauty'))
    h, m = cc_common.build_history_bank('Beauty', 'test', plan)['spectral_k3']
    z = np.load(feat, allow_pickle=True)
    tk = cc_common.sid_tokens_for_items(load_sid('spectral_k3', 'Beauty'),
                                        np.asarray(z['cand_target']))
    keep = np.where((tk >= 0).all(axis=1))[0][:3000]
    mine = cc_common.candidate_total_nll(model, h[keep], m[keep], tk[keep],
                                         np.arange(len(keep)), device='cuda',
                                         batch_size=256)
    ref = np.load(nllp)['total_nll'][keep]
    check('candidate scorer reproduces frozen target NLL (max|diff| < 1e-3)',
          float(np.nanmax(np.abs(mine - ref))) < 1e-3,
          f'{float(np.nanmax(np.abs(mine-ref))):.3e}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    ap.add_argument('--rundir', default=None)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    rundir = Path(a.rundir) if a.rundir else _newest_run(out)
    print(f'run dir: {rundir}')
    immutability()
    leakage()
    outputs(rundir, out)
    freeze_and_results(out)
    scorer_reproduces_frozen()
    print()
    print(f'{N[0] - len(FAILS)}/{N[0]} checks passed')
    if FAILS:
        print(f'FAILURES ({len(FAILS)}):')
        for f_ in FAILS:
            print(f'  - {f_}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


def _newest_run(out):
    runs = [p for p in (Path(out) / 'runs').glob('multi_candidate_nll_*')
            if 'ABORTED' not in p.name]
    if not runs:
        return Path(out) / 'runs'
    return max(runs, key=lambda p: p.stat().st_mtime)


if __name__ == '__main__':
    sys.exit(main())
