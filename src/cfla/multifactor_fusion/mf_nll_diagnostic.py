#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likelihood discriminativeness diagnostic (used by the final report, section E).

For every user whose target is inside the union, measure the fraction of that
user's NON-target candidates that a signal ranks below the target.  0.5 means the
signal carries no information about which candidate is the target; 1.0 means the
target is always first.  Reported for each spectral factorization's teacher-forced
TotalNLL and for the fused retrieval probability, so the report can state whether a
weak likelihood term is a property of the estimator or of the aggregation.

This is a DIAGNOSTIC: it reads the target, is never an input to any method, and is
written after the freeze manifest.

Usage
  python3 mf_nll_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import DATASETS, FLABEL, OUT, S4  # noqa: E402
from mf_core import load_split, z_of  # noqa: E402
from mf_stage import load_scalers, load_targets  # noqa: E402


def rank_fraction(vals, groups, tpos):
    """mean over users of P(non-target candidate has a HIGHER NLL than the target)."""
    fr = []
    for u, t in tpos.items():
        v = vals[groups[u]]
        fr.append(float((v[t] < np.delete(v, t)).mean()))
    return float(np.mean(fr)), len(fr)


def main():
    rows = []
    for split in ('valid', 'test'):
        for ds in DATASETS:
            df, _ = load_split(OUT, ds, 'spectral4', split)
            df.attrs['scalers'] = load_scalers()[ds]
            target = load_targets(ds, split)
            g = df.groupby('user').indices
            tpos = {}
            for u, idx in g.items():
                w = np.where(df.candidate_item.values[idx] == target[u - 1])[0]
                if len(w):
                    tpos[u] = int(w[0])
            P = np.stack([df[f'prob_{FLABEL[f]}'].values for f in S4], 1)
            for f in S4:
                m, n = rank_fraction(df[f'TotalNLL_{FLABEL[f]}'].values, g, tpos)
                rows.append({'split': split, 'dataset': ds, 'signal': f,
                             'rank_frac_vs_nontarget': m, 'n_users': n,
                             'n_candidates': len(df)})
            m, n = rank_fraction(P.mean(1), g, tpos)
            rows.append({'split': split, 'dataset': ds, 'signal': 'Uniform_prob',
                         'rank_frac_vs_nontarget': m, 'n_users': n,
                         'n_candidates': len(df)})
            # z-scored NLL, the quantity the methods actually consume
            Z = z_of(df, S4)
            for j, f in enumerate(S4):
                m, n = rank_fraction(Z[:, j], g, tpos)
                rows.append({'split': split, 'dataset': ds, 'signal': f'zNLL_{f}',
                             'rank_frac_vs_nontarget': m, 'n_users': n,
                             'n_candidates': len(df)})
            print(f'  {split:5s} {ds:32s} written', flush=True)
    out = pd.DataFrame(rows)
    p = OUT / 'validation' / 'nll_discriminativeness.csv'
    out.to_csv(p, index=False)
    print(f'wrote {p} ({len(out)} rows)')
    piv = out[out.split == 'valid'].pivot(index='signal', columns='dataset',
                                          values='rank_frac_vs_nontarget')
    print(piv.round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
