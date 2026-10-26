#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE C — candidate unions from generation output only.

    Pool A  C_S4(u) = C_K2 ∪ C_K3 ∪ C_K4 ∪ C_K5        ("Spectral-4")
    Pool B  C_M5(u) = C_S4(u) ∪ C_RQVAE(u)             ("MultiFactor-5")

Both are built exclusively from the five frozen beam dumps, at the historical
beam budget of 30 per factorization (never increased).  The target is NOT an
input to this script: it is read afterwards, in the separate coverage pass, and
is never added to a union (prompt section 16).

Per (dataset, split) it writes
    <outdir>/<pool>_candidate_union.parquet      one row per (user, candidate)
    <outdir>/<pool>_union_stats.csv              size percentiles + pairwise overlap
and a pooled `_UNION_PROVENANCE.json` with every input hash.

Usage
  python3 mf_build_union.py --split valid
  python3 mf_build_union.py --split test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (DATASETS, FACTORIZATIONS, FLABEL, M5, OUT, S4,  # noqa: E402
                       build_union, load_beam_cell, sha256_file, split_user_ids,
                       write_json)

POOLS = {'spectral4': S4, 'multi5': M5}
STAT_COLS = ['dataset', 'split', 'pool', 'n_users', 'mean_size', 'median_size',
             'p95_size', 'max_size', 'min_size', 'mean_pairs_per_user',
             'n_pairs']


def pairwise_overlap(union, pool):
    """Fraction of users whose target-free candidate sets overlap, by factorization.

    Reported as the Jaccard of each factorization's candidate set against the pool
    union, averaged over users, plus the pairwise factorization-vs-factorization
    Jaccard.  Purely generation-side: no target involved.
    """
    fs = list(pool)
    fi = {f: i for i, f in enumerate(union['f_order'])}
    present = union['present']            # (n, U, F_all)
    n = present.shape[0]
    out = {}
    # each factorization vs the pool union
    for f in fs:
        col = present[:, :, fi[f]]
        inter = col.sum(axis=1).astype(np.float64)
        union_sz = (present[:, :, [fi[x] for x in fs]].any(axis=2)).sum(axis=1)
        j = inter / np.maximum(union_sz, 1)
        out[f'jaccard_vs_union_{f}'] = float(j.mean())
    # pairwise
    for i, a in enumerate(fs):
        for b in fs[i + 1:]:
            ca = present[:, :, fi[a]]
            cb = present[:, :, fi[b]]
            inter = (ca & cb).sum(axis=1).astype(np.float64)
            uni = (ca | cb).sum(axis=1).astype(np.float64)
            out[f'jaccard_{a}_vs_{b}'] = float((inter / np.maximum(uni, 1)).mean())
    return out


def present_count_distribution(union, pool):
    fi = {f: i for i, f in enumerate(union['f_order'])}
    p = union['present'][:, :, [fi[f] for f in pool]]
    cnt = p.sum(axis=2)
    valid = union['items'] >= 0
    cnt = np.where(valid, cnt, -1)
    hist = {}
    for c in range(len(pool) + 1):
        hist[f'candidates_in_{c}_factorisations'] = int((cnt == c).sum())
    return hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--no-parquet', action='store_true',
                    help='write .npy sidecar instead of parquet (debug)')
    ap.add_argument('--stats-only', action='store_true',
                    help='re-derive stats/provenance from existing parquet files')
    a = ap.parse_args(argv)

    outdir = Path(a.outdir) if a.outdir else (OUT / 'validation' if a.split == 'valid'
                                             else OUT / 'test')
    outdir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    stats_rows = []
    prov = {'split': a.split, 'beam': 30, 'pools': {k: list(v) for k, v in POOLS.items()},
            'target_used_in_union_construction': False, 'cells': {}}

    for ds in a.datasets.split(','):
        t0 = time.time()
        cells = {f: load_beam_cell(f, ds, a.split) for f in FACTORIZATIONS}
        users = split_user_ids(ds, a.split)
        union = build_union(cells, user_ids=users)
        n = union['n_users']
        for pool_name, pool in POOLS.items():
            fi = [union['f_order'].index(f) for f in pool]
            # POOL MEMBERSHIP.  The union is built over all five factorizations, so
            # an RQVAE-only candidate exists in `union['items']` even for the
            # spectral4 pool.  A pool contains exactly the candidates that at least
            # one of ITS OWN factorizations generated, so the mask is per pool, not
            # `items >= 0`.
            pool_present = union['present'][:, :, fi].any(axis=2)
            mask = pool_present & (union['items'] >= 0)
            size = mask.sum(axis=1)
            owner = np.repeat(np.arange(n), union['items'].shape[1])
            keep = mask.reshape(-1)
            owner = owner[keep]
            items = union['items'].reshape(-1)[keep]
            # CANONICAL IDS ONLY: `user` and `candidate_item` are both 1-based.
            # The beam dumps store 0-based SID-row indices (docs/ITEM_ID_CONVENTION),
            # so +1 here is the documented conversion; downstream code that needs a
            # SID row does `candidate_item - 1`.
            cols = {'dataset': ds, 'split': a.split,
                    'user': union['user'][owner],          # canonical 1-based user id
                    'user_row': owner,                     # 0-based parquet row index
                    'candidate_item': items + 1}           # canonical 1-based item id
            for f, j in zip(pool, fi):
                cols[f'prob_{FLABEL[f]}'] = union['probs'].reshape(-1, union['probs'].shape[2])[keep, j]
                cols[f'rank_{FLABEL[f]}'] = union['rank'].reshape(-1, union['rank'].shape[2])[keep, j]
                cols[f'present_{FLABEL[f]}'] = union['present'].reshape(-1, union['present'].shape[2])[keep, j]
            cols['n_factorisations_present'] = np.stack(
                [union['present'].reshape(-1, union['present'].shape[2])[keep, j]
                 for j in fi], axis=1).sum(axis=1).astype(np.int8)
            df = pd.DataFrame(cols)
            p = outdir / f'{ds}_{pool_name}_candidate_union.parquet'
            if a.stats_only and p.exists():
                pass
            elif a.no_parquet:
                np.savez_compressed(p.with_suffix('.npz'),
                                    **{k: np.asarray(v) for k, v in cols.items()})
            else:
                df.to_parquet(p, index=False)
            ov = pairwise_overlap(union, pool)
            row = {'dataset': ds, 'split': a.split, 'pool': pool_name, 'n_users': int(n),
                   'mean_size': float(size.mean()), 'median_size': float(np.median(size)),
                   'p95_size': float(np.percentile(size, 95)), 'max_size': int(size.max()),
                   'min_size': int(size.min()),
                   'mean_pairs_per_user': float(len(df) / n), 'n_pairs': int(len(df))}
            row.update(ov)
            stats_rows.append(row)
            prov['cells'][f'{ds}|{pool_name}'] = {
                'n_users': int(n), 'n_pairs': int(len(df)),
                'mean_size': float(size.mean()), 'max_size': int(size.max()),
                'union_parquet': str(p), 'union_sha256': sha256_file(p),
                'inputs': {f: {'beam': str(cells[f]['path']),
                               'sha256': cells[f]['sha256'],
                               'beam_source': ('shared_model_beam' if f == 'rqvae3'
                                               else 'independent_beam'),
                               'raw_item_id_shift': cells[f]['raw_item_id_shift'],
                               'sid_len': cells[f]['sid_len']} for f in FACTORIZATIONS},
                'user_ids_sha256': hashlib.sha256(users.tobytes()).hexdigest(),
                'user_id_convention': '1-based canonical `user` column of the split parquet',
                'present_count_histogram': present_count_distribution(union, pool),
                'elapsed_s': round(time.time() - t0, 1)}
            print(f'{ds:32s} {pool_name:9s} users={n} mean={size.mean():.2f} '
                  f'p95={np.percentile(size, 95):.0f} max={size.max()} '
                  f'({time.time()-t0:.1f}s)', flush=True)
        del union, cells

    extra = []
    for r in stats_rows:
        for k in r:
            if k not in STAT_COLS and k not in extra:
                extra.append(k)
    cols = STAT_COLS + extra
    sp = outdir / f'{a.split}_union_stats.csv'
    with sp.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(stats_rows)
    write_json(outdir / f'_{a.split.upper()}_UNION_PROVENANCE.json', prov)
    print(f'\nwrote {sp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
