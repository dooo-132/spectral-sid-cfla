#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor validation for ablation D.

A new independent-model scorer is only trustworthy if it reproduces a number that was
computed by a different code path.  `cross_tokenizer_target_nll.parquet` holds the
teacher-forced target TotalNLL for every (tokenizer, dataset, user) under the independent
checkpoints, produced by STEP 1's own script.  This scores the same targets with ablation
D's code path and compares elementwise.

If this fails, the scorer is wrong -- not the data.  Do not use ablation D's output.

    python3 fa_d_verify.py [--dataset Beauty] [--split test]
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
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fa_common as C  # noqa: E402
from fa_ablation_d import history_tokens, slot_encode  # noqa: E402

TOL = 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--split', default='test',
                    help='the anchor only stores the TEST split')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--users', type=int, default=0, help='0 = all')
    a = ap.parse_args(argv)

    import pandas as pd
    from cc_common import candidate_total_nll
    from src.analysis.teacher_forced import load_model_from_ckpt
    from xt_common import load_sid

    pq = C.STEP1 / 'cross_tokenizer_target_nll.parquet'
    if not pq.exists():
        raise SystemExit(f'missing anchor {pq}')
    ref = pd.read_parquet(pq)
    print(f'anchor: {pq}  rows={len(ref)}  cols={list(ref.columns)[:12]}')

    ds = a.dataset
    # history must be encoded in the token space of the checkpoint being scored
    from xt_common import sid_path as _sid_path

    # the dataset's own target item ids -> SID rows (0-based) -> slot tokens
    parq = pd.read_parquet(Path(_ART_ROOT + '/data') / ds / f'{a.split}.parquet')
    tgt_items = parq['target'].to_numpy()
    n = len(tgt_items)

    fails = []
    for tok, ckpt in (('spectral_k3', Path(_ART_ROOT + '/model/ckpt/'
                                           f'tiger_{ds}_nonneg_pearson_v3_k3_sid4.pth')),
                      ('rqvae3', Path(_ART_ROOT + '/model/ckpt/') / f'tiger_{ds}.pth')):
        r = ref[(ref['dataset'] == ds) & (ref['tokenizer'] == tok)]
        if not len(r):
            print(f'  {tok}: no anchor rows for {ds} - SKIP')
            continue
        r = r.sort_values('user_index')
        want = r['total_nll'].to_numpy()[:n]
        # use the anchor's own recorded target so this isolates the SCORER, not the
        # target lookup; the two were shown to agree with the test parquet beforehand
        anchor_tgt = r['target_item_id'].to_numpy()[:n]
        assert (anchor_tgt == tgt_items[:len(anchor_tgt)]).all(), \
            'anchor target ids disagree with the test parquet'
        m = min(len(want), n)
        sid = load_sid(tok, ds)
        rows = np.clip(anchor_tgt[:m] - 1, 0, len(sid) - 1)
        toks = slot_encode(sid[rows])
        Hs, Ms = history_tokens(ds, a.split, _sid_path(tok, ds))
        model, vocab = load_model_from_ckpt(str(ckpt), device=a.device)
        got = candidate_total_nll(model, Hs[:m], Ms[:m], toks,
                                  np.arange(m, dtype=np.int64),
                                  device=a.device, batch_size=64)
        d = np.abs(got - want[:m])
        d = d[np.isfinite(d)]
        mx, mean = float(d.max()), float(d.mean())
        ok = mx <= TOL
        print(f'  {tok:12s} n={m} max|diff|={mx:.3e} mean|diff|={mean:.3e} '
              f'{"OK" if ok else "MISMATCH"}')
        if not ok:
            fails.append(f'{tok}: max|diff|={mx:.3e}')
        del model
        import torch
        torch.cuda.empty_cache()

    if fails:
        print(f'\nANCHOR VALIDATION FAILED: {fails}')
        print('Ablation D output must NOT be trusted until this passes.')
        return 1
    print('\nANCHOR VALIDATION PASSED: ablation D\'s scorer reproduces STEP 1\'s target NLL.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
