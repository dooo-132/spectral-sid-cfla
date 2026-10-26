#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION D — shared vs independent models, implemented from scratch.

This is the repair of a delegated attempt that failed before producing results.  It does
not reuse the abandoned file (preserved under `_failed_attempt_backup/` and in git); it
reuses only PROVEN, already-audited code paths:

  * `xt_common.load_cell`            independent beam dumps, with the historical
                                     0-based/1-based item-id shift detected from data
  * `src.analysis.teacher_forced.load_model_from_ckpt`
  * `cc_common.candidate_total_nll`  the batched, target-free candidate scorer used by the
                                     frozen round (model-agnostic: it only needs `.model`)

Question: does candidate-likelihood reranking need the SHARED T5, or does it work with two
independently trained single-factorization models?

The comparison of interest is the WITHIN-REGIME gain (LSE - Uniform), never raw NDCG: the
shared model may simply be a better single-factorization model.

Usage
  python3 fa_ablation_d.py --dataset Beauty          # one dataset (resumable)
  python3 fa_ablation_d.py --dataset all
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
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fa_common as C  # noqa: E402

XT = C.REPO / 'analysis' / 'mechanism' / 'cross_tokenizer_shared'
CC = C.REPO / 'analysis' / 'mechanism' / 'candidate_conditioned'
for _p in (str(XT), str(CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(C.REPO) not in sys.path:
    sys.path.insert(0, str(C.REPO))

CODEBOOK = 256
L = 4
SPECTRAL_BEAM_DIR = Path(_ART_ROOT + '/model/beam_dumps_{ds}_nonneg_pearson_v3')
STEP1_BEAMS = C.STEP1 / 'beams'


def spectral_beam(ds, split):
    return Path(str(SPECTRAL_BEAM_DIR).format(ds=ds)) / f'{split}_k3_sid4.npz'


def rqvae_beam(ds, split):
    return STEP1_BEAMS / f'{ds}_rqvae3_{split}.npz'


def slot_encode(codes):
    """token = code + position*codebook + 1   (docs/SID_FORMAT.md)."""
    pos = np.arange(codes.shape[-1])[None, :]
    return codes + pos * CODEBOOK + 1


def history_tokens(ds, split, sid_p=None):
    """Encoder input per user, in dataset order, via the canonical dataloader.

    `sid_p` MUST be the SID array of the tokenizer whose checkpoint will consume this
    history: GenRecDataset encodes history item ids into that tokenizer's SID tokens.  An
    earlier version always passed the spectral array, which silently fed spectral-token
    histories to the RQ-VAE model; the anchor check caught it (spectral matched exactly,
    rqvae3 did not).
    """
    from src.tiger.dataset import GenRecDataset
    from src.tiger.dataloader import GenRecDataLoader
    cp = Path(sid_p) if sid_p is not None else spectral_sid_path(ds)
    d = GenRecDataset(dataset_path=str(Path(_ART_ROOT + '/data') / ds /
                                      f'{split}.parquet'),
                      code_path=str(cp), mode='evaluation',
                      max_len=20, codebook_size=CODEBOOK)
    H, M = [], []
    for b in GenRecDataLoader(d, batch_size=256, shuffle=False):
        hb = b['history'].numpy()
        H.append(hb); M.append((hb != 0).astype(np.int64))
    return np.concatenate(H).astype(np.int64), np.concatenate(M).astype(np.int64)


def spectral_sid_path(ds):
    from xt_common import sid_path
    return sid_path('spectral_k3', ds)


def rqvae_sid_path(ds):
    from xt_common import sid_path
    return sid_path('rqvae3', ds)


def load_independent_sids(ds):
    from xt_common import load_sid
    return load_sid('spectral_k3', ds), load_sid('rqvae3', ds)


def build_union(a_items, b_items):
    """Per-user union of two (n,30) item-id beams, deduped, -1 dropped.

    Returns (items, owner, in_a, in_b, p_a, p_b) flattened by user.
    """
    n, k = a_items.shape
    items, owner, in_a, in_b, pa, pb = [], [], [], [], [], []
    for u in range(n):
        seen = {}
        for j in range(k):
            it = int(a_items[u, j])
            if it < 0:
                continue
            if it not in seen:
                seen[it] = len(items)
                items.append(it); owner.append(u)
                in_a.append(True); in_b.append(False)
                pa.append(0.0); pb.append(0.0)
            idx = seen[it]
            pa[idx] = 1.0; in_a[idx] = True
        for j in range(k):
            it = int(b_items[u, j])
            if it < 0:
                continue
            if it not in seen:
                seen[it] = len(items)
                items.append(it); owner.append(u)
                in_a.append(False); in_b.append(True)
                pa.append(0.0); pb.append(0.0)
            idx = seen[it]
            pb[idx] = 1.0; in_b[idx] = True
    return (np.asarray(items, np.int64), np.asarray(owner, np.int64),
            np.asarray(in_a, bool), np.asarray(in_b, bool),
            np.asarray(pa), np.asarray(pb))


def union_probs(cell_a, cell_b, items, owner):
    """Beam probability of each union member under each model (0 if outside its beam)."""
    pa = np.zeros(len(items)); pb = np.zeros(len(items))
    ia = cell_a['items']; ib = cell_b['items']
    pr_a = cell_a['probs']; pr_b = cell_b['probs']
    for c in range(len(items)):
        u, it = int(owner[c]), int(items[c])
        ja = np.where(ia[u] == it)[0]
        if len(ja):
            pa[c] = pr_a[u, ja[0]]
        jb = np.where(ib[u] == it)[0]
        if len(jb):
            pb[c] = pr_b[u, jb[0]]
    return pa, pb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_users', type=int, default=64)
    ap.add_argument('--limit_users', type=int, default=0, help='debug: truncate')
    a = ap.parse_args(argv)

    import torch
    from xt_common import load_cell
    from cc_common import candidate_total_nll
    from src.analysis.teacher_forced import load_model_from_ckpt

    datasets = C.DATASETS if a.dataset == 'all' else [a.dataset]
    all_rows, all_val, notes = [], [], []

    for ds in datasets:
        t0 = time.time()
        n_users = C.N_ITEMS[ds]
        print(f'=== {ds} ===', flush=True)
        sidA, sidB = load_independent_sids(ds)
        ckA = Path(_ART_ROOT + '/model/ckpt/'
                   f'tiger_{ds}_nonneg_pearson_v3_k3_sid4.pth')
        ckB = Path(_ART_ROOT + '/model/ckpt/') / f'tiger_{ds}.pth'
        for p in (ckA, ckB):
            if not p.exists():
                raise SystemExit(f'missing independent checkpoint {p}')
        print(f'  ckpt A {ckA.name} sha {C.sha256_file(ckA)[:16]}', flush=True)
        print(f'  ckpt B {ckB.name} sha {C.sha256_file(ckB)[:16]}', flush=True)

        modelA, vA = load_model_from_ckpt(str(ckA), device=a.device)
        modelB, vB = load_model_from_ckpt(str(ckB), device=a.device)
        assert vA == L * CODEBOOK + 1 and vB == L * CODEBOOK + 1, (vA, vB)

        scaler, cells = {}, {}
        for split in ('valid', 'test'):
            ca = load_cell('spectral_k3', ds, split, beam_override=spectral_beam(ds, split))
            cb = load_cell('rqvae3', ds, split, beam_override=rqvae_beam(ds, split))
            if a.limit_users:
                ca['items'] = ca['items'][:a.limit_users]; ca['probs'] = ca['probs'][:a.limit_users]
                ca['target'] = ca['target'][:a.limit_users]; ca['n_users'] = a.limit_users
                cb['items'] = cb['items'][:a.limit_users]; cb['probs'] = cb['probs'][:a.limit_users]
                cb['target'] = cb['target'][:a.limit_users]; cb['n_users'] = a.limit_users
            n = ca['n_users']
            HA, MA = history_tokens(ds, split, spectral_sid_path(ds))
            HB, MB = history_tokens(ds, split, rqvae_sid_path(ds))
            if a.limit_users:
                HA, MA = HA[:a.limit_users], MA[:a.limit_users]
                HB, MB = HB[:a.limit_users], MB[:a.limit_users]
            assert len(HA) == n and len(HB) == n, (len(HA), len(HB), n)
            items, owner, in_a, in_b, _, _ = build_union(ca['items'], cb['items'])
            pa, pb = union_probs(ca, cb, items, owner)
            toks = slot_encode(sidA[np.clip(items - 1, 0, len(sidA) - 1)])
            toks_b = slot_encode(sidB[np.clip(items - 1, 0, len(sidB) - 1)])
            # each factorization's candidates are scored against a history encoded in
            # THAT factorization's token space
            n1 = candidate_total_nll(modelA, HA, MA, toks, owner, device=a.device,
                                     batch_size=a.batch_users)
            n2 = candidate_total_nll(modelB, HB, MB, toks_b, owner, device=a.device,
                                     batch_size=a.batch_users)
            cells[split] = dict(items=items, owner=owner, pa=pa, pb=pb, n1=n1, n2=n2,
                                tgt=ca['target'], n_users=n)
            if split == 'valid':
                fin = np.isfinite(n1) & np.isfinite(n2)
                scaler = {'N1': (float(n1[fin].mean()), float(n1[fin].std(ddof=0))),
                          'N2': (float(n2[fin].mean()), float(n2[fin].std(ddof=0)))}
                print(f'  valid scaler mu1={scaler["N1"][0]:.4f} sd1={scaler["N1"][1]:.4f} '
                      f'mu2={scaler["N2"][0]:.4f} sd2={scaler["N2"][1]:.4f}', flush=True)
            print(f'  {split}: users={n} union={len(items)} '
                  f'({len(items)/n:.2f}/user) in {time.time()-t0:.0f}s', flush=True)
            del HA, MA, HB, MB

        # ---- evaluate TEST ----
        c = cells['test']
        g = {'users': np.arange(c['n_users']), 'item': c['items'],
             'p1': c['pa'], 'p2': c['pb'],
             'N1': c['n1'], 'N2': c['n2'],
             # starts[u] = first row with owner >= u ; ends[u] = first row with owner >= u+1.
             # Using arange(n+1) for ends makes ends[u] == starts[u], i.e. empty segments,
             # which silently zeroes coverage.
             'starts': np.searchsorted(c['owner'], np.arange(c['n_users'])),
             'ends': np.searchsorted(c['owner'], np.arange(1, c['n_users'] + 1))}
        tgt = c['tgt']
        # target as a within-segment offset (same convention as the rest of the suite)
        idx = np.full(c['n_users'], -1, dtype=np.int64)
        for u in range(c['n_users']):
            t = int(tgt[u])
            if t < 0:
                continue
            hit = np.where(c['items'][g['starts'][u]:g['ends'][u]] == t)[0]
            if len(hit):
                idx[u] = int(hit[0])
        uni = C.score(g, 'Uniform', scaler)
        lse = C.score(g, 'ZLSE', scaler)
        r_uni = C.rank_of_target_rel(g, uni, idx)
        r_lse = C.rank_of_target_rel(g, lse, idx)
        nd_u = C.ndcg_from_rank(r_uni, 20); nd_l = C.ndcg_from_rank(r_lse, 20)
        boot = C.paired_bootstrap(nd_l, nd_u)
        cov = float((idx >= 0).mean())
        all_rows.append({'dataset': ds, 'regime': 'independent', 'method': 'Uniform_ind',
                         'n_users': c['n_users'], 'coverage': cov,
                         'mean_union_size': len(c['items']) / c['n_users'],
                         'NDCG@20': float(nd_u.mean()),
                         'Recall@20': float(C.recall_from_rank(r_uni, 20).mean()),
                         'mu_F1': scaler['N1'][0], 'sigma_F1': scaler['N1'][1],
                         'mu_F2': scaler['N2'][0], 'sigma_F2': scaler['N2'][1]})
        all_rows.append({'dataset': ds, 'regime': 'independent', 'method': 'ZLSE_ind',
                         'n_users': c['n_users'], 'coverage': cov,
                         'mean_union_size': len(c['items']) / c['n_users'],
                         'NDCG@20': float(nd_l.mean()),
                         'Recall@20': float(C.recall_from_rank(r_lse, 20).mean()),
                         'mu_F1': scaler['N1'][0], 'sigma_F1': scaler['N1'][1],
                         'mu_F2': scaler['N2'][0], 'sigma_F2': scaler['N2'][1]})
        all_val.extend([
            {'comparison': 'LSE_ind_vs_Uniform_ind', 'dataset': ds, 'metric': 'NDCG@20',
             **boot}])
        print(f'  TEST Uniform={nd_u.mean():.6f} LSE={nd_l.mean():.6f} '
              f'gain={nd_l.mean()-nd_u.mean():+.6f} coverage={cov:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        del modelA, modelB
        torch.cuda.empty_cache()

    if a.limit_users:
        print('LIMIT_USERS set: not writing outputs'); return 0

    C.write_csv(C.OUT / 'shared_vs_independent.csv', all_rows)
    C.write_csv(C.OUT / 'shared_vs_independent_bootstrap.csv', all_val)
    print(f'wrote shared_vs_independent.csv ({len(all_rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
