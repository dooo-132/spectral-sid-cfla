#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART L — efficiency accounting for the disagreement gate.

The gate does not remove the candidate-NLL cost: it consumes the same frozen NLL and only
changes the mixing weight.  This measures what it actually costs and states plainly that
there is no compute saving.

Run:  python3 dg_efficiency.py --outdir outputs/disagreement_gate
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
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dg_features import DATASETS, SHORT  # noqa: E402
from dg_eval import build, load_winner  # noqa: E402


def timeit(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    (out / 'efficiency').mkdir(parents=True, exist_ok=True)
    tag, model, ck = load_winner(out)
    params = int(sum(p.numel() for p in model.parameters()))
    rows = [{'metric': 'gate_parameters', 'scope': tag, 'value': params, 'unit': 'params',
             'note': f'{ck["kind"]}, lambda_g={ck["lambda_g"]}, d_in={ck["d_in"]}'}]

    PT, _ = build(out, 'test')
    for ds in DATASETS:
        p = PT[ds]
        sizes = p.ends - p.starts
        rows.append({'metric': 'candidate_union_size_mean', 'scope': SHORT[ds],
                     'value': f'{sizes.mean():.2f}', 'unit': 'candidates/user',
                     'note': 'unchanged from the previous round; the gate does not alter it'})
        rows.append({'metric': 'nll_sequences_per_user', 'scope': SHORT[ds],
                     'value': f'{sizes.mean() * 2:.1f}', 'unit': 'sequences/user',
                     'note': '2 factorizations x union size; REQUIRED, not saved'})

    # gate forward latency, batch 1 and batch 256 users
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    p = PT['Beauty']
    for bs in (1, 256):
        W = p.W; d_in = p.X.shape[2]
        x = torch.randn(bs, W, d_in, device=dev)
        with torch.no_grad():
            t = timeit(lambda: model(x), repeats=5, warmup=2)
        rows.append({'metric': f'gate_latency_batch{bs}_users', 'scope': tag,
                     'value': f'{t*1000:.3f}', 'unit': 'ms',
                     'note': f'{bs} users x {W} candidates on {dev}'})

    # mixing arithmetic only
    n = len(p.ZU_flat)
    zu = np.random.default_rng(0).random(n); zl = np.random.default_rng(1).random(n)
    g = np.random.default_rng(2).random(n)
    t = timeit(lambda: (1 - g) * zu + g * zl, repeats=5, warmup=2)
    rows.append({'metric': 'mixing_latency_full_split', 'scope': f'Beauty {n} candidates',
                 'value': f'{t*1000:.3f}', 'unit': 'ms',
                 'note': 'S = (1-g)Z_U + gZ_L, vectorised, no model'})

    # inherited cost of producing the NLL that the gate consumes
    rows.append({'metric': 'candidate_nll_cost_inherited', 'scope': 'all datasets',
                 'value': 'not recomputed this round', 'unit': '-',
                 'note': 'reused the audited parquet artifacts; see '
                         'CANDIDATE_NLL_REUSE_AUDIT.md'})
    rows.append({'metric': 'inference_compute_saving', 'scope': tag, 'value': '0',
                 'unit': 'fraction',
                 'note': 'the gate consumes the same per-candidate NLL; it saves nothing'})

    with (out / 'efficiency.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['metric', 'scope', 'value', 'unit', 'note'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({'gate': tag, 'params': params,
               'saves_compute': False,
               'why': 'per-candidate NLL is still required for every candidate; the gate '
                      'only reweights the frozen scores'},
              open(out / 'efficiency' / 'EFFICIENCY.json', 'w'), indent=2)
    print(f'wrote {out/"efficiency.csv"} ({len(rows)} rows)')
    for r in rows:
        print(f'  {r["metric"]:34s} {r["scope"][:26]:26s} {str(r["value"]):>14s} {r["unit"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
