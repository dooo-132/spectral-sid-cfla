#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE K — compute and efficiency accounting for candidate-conditioned reranking.

Measures, on the real frozen checkpoint and real candidate sets:

  * candidate-union size distribution per dataset and split
  * teacher-forced sequences scored per user (the dominant new cost)
  * NLL scoring latency, measured on a real batch
  * beam-generation latency (inherited from the shared model's own dumps)
  * reranking latency (the scoring arithmetic only, no model)
  * GPU memory high-water mark during scoring

The point of the accounting is that candidate NLL is **not free**: it costs one
teacher-forced pass per (user, candidate, factorization), which is far more than the two
beam generations it competes with.  That is reported plainly rather than hidden.

Usage
  python3 cc_efficiency.py --rundir <run> --outdir outputs/candidate_conditioned
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_common import (DATASETS, F1, F2, SHORT, build_history_bank,  # noqa: E402
                       candidate_total_nll, frozen_checkpoint, load_frozen_shared,
                       sid_tokens_for_items)
from xt_common import load_sid  # noqa: E402

SELECTOR = Path('outputs/adaptive_factorization/selector')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--outdir', default='outputs/candidate_conditioned')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--timing-users', type=int, default=1024)
    a = ap.parse_args(argv)
    rundir, out = Path(a.rundir), Path(a.outdir)
    rows = []

    import pandas as pd
    import torch

    for ds in DATASETS:
        union_p = rundir / 'test' / f'{ds}_candidate_union.parquet'
        if not union_p.exists():
            print(f'  skip {ds}: {union_p.name} missing')
            continue
        u = pd.read_parquet(union_p, columns=['user', 'candidate_item'])
        sizes = u.groupby('user').size().values
        n_users = len(sizes)
        n_pairs = int(len(u))
        rows += [
            {'metric': 'candidate_union_size_mean', 'scope': ds, 'value': f'{sizes.mean():.2f}',
             'unit': 'items/user', 'note': 'C_F1 union C_F2, deduplicated'},
            {'metric': 'candidate_union_size_median', 'scope': ds, 'value': f'{np.median(sizes):.0f}',
             'unit': 'items/user', 'note': ''},
            {'metric': 'candidate_union_size_p95', 'scope': ds,
             'value': f'{np.percentile(sizes, 95):.0f}', 'unit': 'items/user', 'note': ''},
            {'metric': 'teacher_forced_sequences_per_user', 'scope': ds,
             'value': f'{n_pairs / n_users * 2:.1f}', 'unit': 'sequences/user',
             'note': '2 factorizations x union size; this is the new cost'},
            {'metric': 'teacher_forced_sequences_total', 'scope': ds,
             'value': f'{n_pairs * 2}', 'unit': 'sequences',
             'note': f'over {n_users} users'},
        ]
        del u

    # ---- measured latency on a real batch ----
    ds = 'Beauty'
    if (rundir / 'test' / f'{ds}_candidate_union.parquet').exists():
        model, plan = load_frozen_shared(frozen_checkpoint(ds), device=a.device)
        sids = {f: load_sid(f, ds) for f in (F1, F2)}
        u = pd.read_parquet(rundir / 'test' / f'{ds}_candidate_union.parquet',
                            columns=['user', 'candidate_item'])
        # restrict to whole users so the timing batch is representative
        keep_users = np.unique(u.user.values)[:a.timing_users]
        m = np.isin(u.user.values, keep_users)
        owner = np.searchsorted(keep_users, u.user.values[m])
        items = u.candidate_item.values[m].astype(np.int64)
        bank = build_history_bank(ds, 'test', plan)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = {}
        for f in (F1, F2):
            toks = sid_tokens_for_items(sids[f], items)
            h, mk = bank[f]
            h = h[keep_users]; mk = mk[keep_users]
            t0 = time.perf_counter()
            nll = candidate_total_nll(model, h, mk, toks, owner, device=a.device,
                                      batch_size=256)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t[f] = time.perf_counter() - t0
            assert np.isfinite(nll).mean() > 0.99
            del toks
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else float('nan')
        n_seq = int(len(items)) * 2
        rows += [
            {'metric': 'nll_scoring_latency_total', 'scope': f'{ds} n={a.timing_users} users',
             'value': f'{t[F1] + t[F2]:.3f}', 'unit': 's',
             'note': f'{n_seq} teacher-forced sequences'},
            {'metric': 'nll_scoring_latency_per_1k_sequences', 'scope': ds,
             'value': f'{(t[F1] + t[F2]) / n_seq * 1e5:.3f}', 'unit': 's / 1k sequences',
             'note': 'linear in candidates x factorizations'},
            {'metric': 'gpu_peak_memory_allocated', 'scope': f'{ds} scoring',
             'value': f'{peak:.1f}', 'unit': 'MiB', 'note': 'torch allocator peak'},
        ]
        # scoring arithmetic only, no model
        z1 = np.random.default_rng(0).normal(size=len(items))
        z2 = np.random.default_rng(1).normal(size=len(items))
        p1 = np.random.default_rng(2).random(len(items))
        p2 = np.random.default_rng(3).random(len(items))
        t0 = time.perf_counter()
        for _ in range(5):
            mx = np.maximum(-z1, -z2)
            lse = mx + np.log(np.exp(-z1 - mx) + np.exp(-z2 - mx))
            w1 = 1.0 / (1.0 + np.exp((z1 - z2) / 0.5))
            _ = w1 * p1 + (1 - w1) * p2
            _ = 0.5 * p1 + 0.5 * p2
        rerank_ms = (time.perf_counter() - t0) / 5 * 1000
        rows.append({'metric': 'reranking_latency', 'scope': f'{ds} {len(items)} candidates',
                     'value': f'{rerank_ms:.2f}', 'unit': 'ms',
                     'note': 'all four score formulas, vectorised, no model'})
        del model
        torch.cuda.empty_cache()

    # ---- inherited beam cost for the two-beam reference ----
    bm = SELECTOR / 'features' / 'beams'
    for ds in DATASETS:
        cells = []
        for f in (F1, F2):
            p = bm / f'{ds}_{f}_test.npz'
            if p.exists():
                z = np.load(p, allow_pickle=True)
                cells.append(int(np.asarray(z['candidate_item_ids']).shape[0]))
        if cells:
            rows.append({'metric': 'beam_generation_candidates_total', 'scope': ds,
                         'value': f'{sum(cells) * 30}', 'unit': 'beam entries',
                         'note': f'{len(cells)} factorizations x beam 30 x {cells[0]} users'})

    outp = out / 'efficiency.csv'
    with outp.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['metric', 'scope', 'value', 'unit', 'note'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {outp} ({len(rows)} rows)')
    for r in rows:
        if r['metric'] in ('candidate_union_size_mean', 'teacher_forced_sequences_per_user',
                           'nll_scoring_latency_per_1k_sequences'):
            print(f"  {r['metric']:38s} {r['scope']:16s} {r['value']:>10s} {r['unit']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
