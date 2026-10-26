#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify — from disk — that this suite's reimplementation of the frozen method reproduces
the published numbers, for all five datasets.

This is the gate for everything else: if Uniform/LSE do not reproduce the frozen
artefacts, no ablation built on this code can be trusted.  Read-only.

    python3 fa_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fa_common as C  # noqa: E402

TOL = 5e-7          # published values are rounded to 6 dp


def published_dev():
    """Frozen DEV test NDCG@20, read from the disagreement-gate round's own output."""
    import csv
    rows = list(csv.DictReader((C.DG / 'test' / 'test_results.csv').open()))
    out = {}
    for r in rows:
        out[(r['dataset'], r['method'])] = float(r['NDCG@20'])
    return out


def published_sports():
    import csv
    rows = list(csv.DictReader((C.SPORTS / 'test' / 'test_results.csv').open()))
    return {r['method']: float(r['NDCG@20']) for r in rows}


def main():
    fails = []
    print('=' * 78)
    print('VERIFY: reimplementation vs frozen published numbers')
    print('=' * 78)

    dev_pub = published_dev()
    print('\n--- development datasets (frozen scaler, frozen candidate parquet) ---')
    print(f'{"dataset":12s} {"method":8s} {"recomputed":>12s} {"published":>12s} {"delta":>10s}')
    for ds in C.DATASETS:
        sc = C.scaler_from_json(C.CANON_SCALER, datasets_key=ds)
        u, n = C.load_cells_dev(ds, 'test')
        g = C.grouped_arrays(u, n)
        tgt = C.load_targets_dev(ds, 'test')
        C.assert_targets_align(g, tgt, n_items=C.N_ITEMS[ds])
        for method, key in (('Uniform', 'Uniform'), ('ZLSE', 'LSE')):
            r = C.rank_of_target(g, C.score(g, method, sc), tgt)
            got = float(C.ndcg_from_rank(r, 20).mean())
            want = dev_pub[(ds, key)]
            d = got - want
            flag = '' if abs(d) <= TOL else '   <-- MISMATCH'
            if abs(d) > TOL:
                fails.append(f'{ds}/{key}: {got:.8f} != {want:.8f}')
            print(f'{C.SHORT[ds]:12s} {key:8s} {got:12.8f} {want:12.8f} {d:+10.2e}{flag}')
        del u, n, g

    print('\n--- Sports (pristine confirmation artefacts, frozen scaler) ---')
    sp_pub = published_sports()
    sc = C.scaler_from_json(C.SPORTS / 'valid' / 'candidate_nll_scaler.json')
    u, n = C.load_cells_sports('test')
    g = C.grouped_arrays_asis(u, n)   # frozen Sports layout: raw row order
    tgt = C.load_targets_sports_test()
    if len(tgt) != len(g['users']):
        fails.append(f'Sports target len {len(tgt)} != n_users {len(g["users"])}')
    # tgt_rel is a WITHIN-segment offset, not a global SID row index
    if (tgt >= 0).any() and tgt.max() < 1000 and len(np.unique(tgt[tgt >= 0])) < 1000:
        print('  (tgt_rel is a within-segment offset; using rank_of_target_rel)')
    for method, key in (('Uniform', 'Uniform'), ('ZLSE', 'LSE')):
        r = C.rank_of_target_rel(g, C.score(g, method, sc), tgt)
        got = float(C.ndcg_from_rank(r, 20).mean())
        want = sp_pub[key]
        d = got - want
        flag = '' if abs(d) <= TOL else '   <-- MISMATCH'
        if abs(d) > TOL:
            fails.append(f'Sports/{key}: {got:.8f} != {want:.8f}')
        print(f'{"Sports":12s} {key:8s} {got:12.8f} {want:12.8f} {d:+10.2e}{flag}')

    # --- scaler provenance: refitting on VALID must reproduce the frozen scaler ---
    print('\n--- scaler reproducibility (refit on VALID ALL candidates) ---')
    for ds in C.DATASETS:
        uv, nv = C.load_cells_dev(ds, 'valid')
        gv = C.grouped_arrays(uv, nv)
        refit = C.fit_scaler(gv)
        froz = C.scaler_from_json(C.CANON_SCALER, datasets_key=ds)
        dm1 = abs(refit['N1'][0] - froz['N1'][0]); ds1 = abs(refit['N1'][1] - froz['N1'][1])
        dm2 = abs(refit['N2'][0] - froz['N2'][0]); ds2 = abs(refit['N2'][1] - froz['N2'][1])
        ok = max(dm1, ds1, dm2, ds2) < 1e-9
        if not ok:
            fails.append(f'{ds}: refit scaler differs from frozen')
        print(f'  {C.SHORT[ds]:12s} mu/sd match frozen: {ok}  '
              f'(max|d| = {max(dm1, ds1, dm2, ds2):.2e})')
        del uv, nv, gv

    print('\n' + '=' * 78)
    if fails:
        print(f'VERIFY FAILED ({len(fails)}):')
        for f in fails:
            print('  -', f)
        return 1
    print('VERIFY OK: reimplementation reproduces every frozen published number.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
