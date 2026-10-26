#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consistency guards for the factorization-generality study.

1. The teacher-forced spectral NLL recomputed here must equal the audited
   `full_id_reaudit/teacher_forced_full_sid_nll.csv` (regression test).
2. Single-model canonical NDCG@20 must equal the audited mechanism-study values.
3. Total identity conditional entropy must equal log N (a mathematical identity
   of an injective SID map) for every family and K.
4. The two-model spectral G-pair fusion NDCG@20 must equal the audited
   per-subset fusion values where the same subset exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import OUT, REPO  # noqa: E402

AUDIT = REPO / 'outputs' / 'mechanism' / 'full_id_reaudit'
FAIL = []


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        FAIL.append(name)


def main():
    # ---- 1. spectral NLL regression -------------------------------------
    new = pd.read_parquet(OUT / 'full_id_nll_all_factorizations.parquet')
    old = pd.read_csv(AUDIT / 'teacher_forced_full_sid_nll.csv')
    new_sp = (new[new['family'] == 'spectral']
              .groupby(['dataset', 'K'])['total_nll'].mean().rename('new'))
    old_i = old.set_index(['dataset', 'K'])['mean_total_nll'].rename('old')
    j = pd.concat([new_sp, old_i], axis=1).dropna()
    diff = (j['new'] - j['old']).abs()
    check('teacher-forced spectral TotalNLL reproduces the audited values',
          bool((diff < 1e-6).all()),
          f'max|diff|={diff.max():.3e} over {len(j)} cells')

    # ---- 2. canonical single-model NDCG@20 ------------------------------
    art = pd.read_csv(OUT / 'ARTIFACT_MATRIX.csv')
    import json
    sp = art[art['family'] == 'spectral'].copy()
    sp['ndcg20'] = sp['metrics'].apply(lambda s: json.loads(s)['NDCG@20'])
    audited = {('Beauty', 2): 0.040734, ('Beauty', 3): 0.039965,
               ('Beauty', 4): 0.039878, ('Beauty', 5): 0.039765}
    ok = True
    det = []
    for (ds, k), v in audited.items():
        got = float(sp[(sp['dataset'] == ds) & (sp['K'] == k)]['ndcg20'].iloc[0])
        ok &= abs(got - v) < 5e-6
        det.append(f'{ds}K{k}:{got:.6f}/{v:.6f}')
    check('canonical Beauty spectral NDCG@20 reproduces the mechanism study',
          ok, ' '.join(det))

    # ---- 3. identity entropy identity ------------------------------------
    ent = pd.read_csv(OUT / 'identity_entropy_all_factorizations.csv')
    summ = ent[ent['position'] == -1]
    d = (summ['identity_entropy_total'] - summ['identity_entropy_log_N']).abs()
    check('sum_t H(C_t|C_<t) == log N for every family x K',
          bool((d < 1e-9).all()), f'max|diff|={d.max():.3e} over {len(summ)} cells')

    # ---- 4. fusion values reported by the mechanism study ------------------
    from fg_common import K_LIST, load_cell, summarize, union_fuse
    cells = {k: load_cell('Beauty', 'spectral', k, 'test') for k in K_LIST}
    checks = [
        ('two-model K2+K3', union_fuse([cells[2], cells[3]]), 0.04395),
        ('three-model K2+K3+K5',
         union_fuse([cells[2], cells[3], cells[5]]), 0.04567),
        ('four-model K2+K3+K4+K5', union_fuse(list(cells.values())), 0.046190),
    ]
    det, ok = [], True
    for name, res, expect in checks:
        got = summarize(res['rank'])['NDCG@20']
        ok &= abs(got - expect) < 3e-5
        det.append(f'{name}:{got:.6f}/{expect:.6f}')
    check('Beauty spectral fusion reproduces the mechanism-study values',
          ok, ' '.join(det))

    # ---- 5. vectorised fusion vs the independent reference implementation --
    import itertools
    from fg_common import K_LIST, load_cell, summarize, union_fuse
    from src.analysis.per_user import build_view, fuse_subset, summarize_ranks
    n_chk = 0
    bad = []
    for ds in ('Beauty', 'Toys_and_Games'):
        v = build_view(ds)
        cells = {k: load_cell(ds, 'spectral', k, 'test') for k in K_LIST}
        for r in (1, 2, 3, 4):
            for sub in itertools.combinations(K_LIST, r):
                ref = summarize_ranks(fuse_subset(v, sub)[1])['NDCG@20']
                mine = summarize(union_fuse([cells[k] for k in sub])['rank'])['NDCG@20']
                n_chk += 1
                if abs(ref - mine) > 1e-9:
                    bad.append((ds, sub, ref, mine))
    check('vectorised union fusion == reference per_user.fuse_subset',
          not bad, f'checked {n_chk} subsets on 2 datasets; mismatches={len(bad)}')

    print()
    if FAIL:
        print('FAILURES:', FAIL)
        return 1
    print('ALL CONSISTENCY CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
