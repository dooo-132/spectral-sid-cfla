#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — target-free factorization router: data assembly, training, RouterDev model
selection, static-weight selection and the freeze manifest.

DESIGN RULES ENFORCED HERE
--------------------------
* Router supervision uses the chronologically EARLIER canonical **validation**
  instance.  The canonical **test** instance is never read.
* Users are split 70/15/15 into RouterTrain / RouterDev / RouterHoldout by a fixed
  seed, at USER level, and the split is written to `router_user_split.csv` once.
* Every feature is target-free.  **No test metric is computed in this script.**
* The scaler is fitted on RouterTrain only.
* Architecture (Linear vs MLP) and feature family (Router H vs Router P) and the
  static fusion weight are selected on RouterDev only, then frozen.

The RouterDev objective is the soft-fusion NDCG@20, computed exactly from the two
factorizations' compact candidate sets (`af_common.fused_rank`), so model selection
evaluates the real deployed scoring rule rather than a surrogate.

Usage
  python3 af_router_train.py --selectordir <selector dir>
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_common import (FusedCandidates, BOOT_SEED, fused_rank, paired_bootstrap,  # noqa: E402
                       per_user_ndcg)

DATASET_ORDER = ['Beauty', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
                 'Toys_and_Games']
SPLIT_SEED = 20260914
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ('RouterTrain', 'RouterDev', 'RouterHoldout')
TOPK_EVAL = 20
STATIC_GRID = [round(0.1 * i, 1) for i in range(11)]


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def load_cells(selectordir, split):
    d = Path(selectordir) / 'features'
    out = {}
    for ds in DATASET_ORDER:
        p = d / f'{ds}_{split}.npz'
        if not p.exists():
            raise FileNotFoundError(p)
        out[ds] = dict(np.load(p, allow_pickle=True))
    return out


def make_split(parts):
    """Deterministic user-level split (same users get the same role in both splits)."""
    rng = np.random.default_rng(SPLIT_SEED)
    assign = {}
    for ds in DATASET_ORDER:
        users = np.asarray(parts[ds]['user'])
        perm = rng.permutation(len(users))
        n = len(users)
        n_tr = int(round(SPLIT_FRACTIONS[0] * n))
        n_dv = int(round(SPLIT_FRACTIONS[1] * n))
        a = np.empty(n, dtype=object)
        a[perm[:n_tr]] = 'RouterTrain'
        a[perm[n_tr:n_tr + n_dv]] = 'RouterDev'
        a[perm[n_tr + n_dv:]] = 'RouterHoldout'
        assign[ds] = a
    return assign


def write_split_csv(path, assign, parts):
    with Path(path).open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'row_index', 'user', 'split'])
        for ds in DATASET_ORDER:
            users = np.asarray(parts[ds]['user'])
            for i in range(len(users)):
                w.writerow([ds, i, int(users[i]), str(assign[ds][i])])


def read_split_csv(path):
    assign = {ds: None for ds in DATASET_ORDER}
    rows = {ds: [] for ds in DATASET_ORDER}
    with Path(path).open() as fh:
        for r in csv.DictReader(fh):
            rows[r['dataset']].append((int(r['row_index']), r['split']))
    for ds in DATASET_ORDER:
        rows[ds].sort()
        assign[ds] = np.array([s for _, s in rows[ds]], dtype=object)
    return assign


def cat(parts, assign, role, key):
    X, q, u, w = [], [], [], []
    for ds in DATASET_ORDER:
        keep = assign[ds] == role
        p = parts[ds]
        X.append(np.asarray(p[key])[keep])
        q.append(np.stack([np.asarray(p['q1'])[keep], np.asarray(p['q2'])[keep]], axis=1))
        u.append(np.stack([np.asarray(p['u1'])[keep], np.asarray(p['u2'])[keep]], axis=1))
        w.append(np.asarray(p['weight'])[keep])
    return (np.concatenate(X), np.concatenate(q), np.concatenate(u), np.concatenate(w))


def cat_candidates(parts, assign, role):
    i1, p1, i2, p2, t = [], [], [], [], []
    for ds in DATASET_ORDER:
        keep = assign[ds] == role
        p = parts[ds]
        i1.append(np.asarray(p['cand_ids1'])[keep]); p1.append(np.asarray(p['cand_p1'])[keep])
        i2.append(np.asarray(p['cand_ids2'])[keep]); p2.append(np.asarray(p['cand_p2'])[keep])
        t.append(np.asarray(p['cand_target'])[keep])
    return FusedCandidates(np.concatenate(i1), np.concatenate(p1),
                           np.concatenate(i2), np.concatenate(p2), np.concatenate(t))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class LinearRouter(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.fc(x)


class MLPRouter(nn.Module):
    def __init__(self, d_in, hidden=64, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(p_drop), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def soft_ce(logits, q, w):
    logp = torch.log_softmax(logits, dim=1)
    per = -(q * logp).sum(dim=1)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def state_hash(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def git_sha():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                           cwd=str(HERE.parents[2]))
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_router(kind, d_in, Xtr, qtr, wtr, dev_score_fn, seed, lr, batch, max_epochs,
                 patience, verbose=True):
    torch.manual_seed(seed)
    model = LinearRouter(d_in) if kind == 'linear' else MLPRouter(d_in)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    qtr_t = torch.tensor(qtr, dtype=torch.float32)
    wtr_t = torch.tensor(wtr, dtype=torch.float32)
    n = Xtr_t.shape[0]
    rng = np.random.default_rng(seed)
    hist, best, bad = [], {'score': -np.inf, 'epoch': -1, 'state': None}, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = torch.as_tensor(perm[s:s + batch], dtype=torch.long)
            opt.zero_grad()
            loss = soft_ce(model(Xtr_t[idx]), qtr_t[idx], wtr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        model.eval()
        score = dev_score_fn(model)
        hist.append({'epoch': ep + 1, 'train_loss': tot / max(nb, 1), 'dev_score': score})
        if verbose and ((ep + 1) % 20 == 0 or ep == 0):
            print(f'    ep{ep+1:3d} loss={tot/max(nb,1):.5f} dev={score:.6f}', flush=True)
        if score > best['score'] + 1e-9:
            best = {'score': score, 'epoch': ep + 1,
                    'state': copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best['state'])
    model.eval()
    return model, best, hist



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selectordir', required=True)
    ap.add_argument('--max_epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=2025)
    a = ap.parse_args(argv)

    sel = Path(a.selectordir)
    for sub in ('features', 'scalers', 'checkpoints'):
        (sel / sub).mkdir(parents=True, exist_ok=True)

    parts = load_cells(sel, 'valid')
    split_csv = sel / 'router_user_split.csv'
    if split_csv.exists():
        assign = read_split_csv(split_csv)
        for ds in DATASET_ORDER:
            if len(assign[ds]) != len(parts[ds]['user']):
                raise AssertionError(f'{split_csv} does not match {ds} row count')
        print(f'reusing existing user split ({split_csv})', flush=True)
    else:
        assign = make_split(parts)
        write_split_csv(split_csv, assign, parts)
        print(f'wrote {split_csv}', flush=True)

    tr = cat(parts, assign, 'RouterTrain', 'router_h')
    dv = cat(parts, assign, 'RouterDev', 'router_h')
    tr_p = cat(parts, assign, 'RouterTrain', 'router_p')
    dv_p = cat(parts, assign, 'RouterDev', 'router_p')
    dev_cand = cat_candidates(parts, assign, 'RouterDev')

    # scaler fitted on RouterTrain only
    scalers = {}
    for key, Xtr in (('router_h', tr[0]), ('router_p', tr_p[0])):
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        scalers[key] = (mu, sd)
        np.savez(sel / 'scalers' / f'{key}_scaler.npz', mean=mu, std=sd,
                 fitted_on=np.array('RouterTrain'), n_fit=np.array(Xtr.shape[0]))
    scaler_hash = hashlib.sha256(
        b''.join(scalers[k][0].tobytes() + scalers[k][1].tobytes()
                 for k in ('router_h', 'router_p'))).hexdigest()

    def z(key, X):
        mu, sd = scalers[key]
        return (X - mu) / sd

    Xtr_h, Xdv_h = z('router_h', tr[0]), z('router_h', dv[0])
    Xtr_p, Xdv_p = z('router_p', tr_p[0]), z('router_p', dv_p[0])

    def dev_ndcg(w1vec):
        return float(per_user_ndcg(fused_rank(w1vec, dev_cand), TOPK_EVAL).mean())

    uniform_dev = dev_ndcg(np.full(len(dev_cand), 0.5))
    print(f'RouterTrain n={Xtr_h.shape[0]}  RouterDev n={Xdv_h.shape[0]}  '
          f'uniform dev NDCG@20={uniform_dev:.6f}', flush=True)

    def make_dev_fn(Xdv_f):
        Xdv_t = torch.tensor(Xdv_f, dtype=torch.float32)

        def fn(model):
            with torch.no_grad():
                w = torch.softmax(model(Xdv_t), dim=1).numpy()
            return dev_ndcg(w[:, 0])
        return fn

    names_h = [str(x) for x in np.asarray(parts[DATASET_ORDER[0]]['feature_names_h'])]
    names_p = [str(x) for x in np.asarray(parts[DATASET_ORDER[0]]['feature_names_p'])]

    results = []
    family_best = {}
    for feat_key, Xtr_f, Xdv_f, names in (('H', Xtr_h, Xdv_h, names_h),
                                          ('P', Xtr_p, Xdv_p, names_p)):
        for kind in ('linear', 'mlp'):
            tag = f'router_{feat_key}_{kind}'
            print(f'  training {tag} (d_in={Xtr_f.shape[1]})', flush=True)
            t0 = time.time()
            model, best, hist = train_router(
                kind, Xtr_f.shape[1], Xtr_f, tr[1], tr[3], make_dev_fn(Xdv_f),
                a.seed, a.lr, a.batch, a.max_epochs, a.patience)
            rec = {'tag': tag, 'feat_key': feat_key, 'kind': kind, 'd_in': int(Xtr_f.shape[1]),
                   'n_params': count_params(model), 'dev_fusion_ndcg20': best['score'],
                   'best_epoch': best['epoch'], 'epochs_run': len(hist),
                   'checkpoint_sha256': state_hash(model),
                   'train_seconds': round(time.time() - t0, 1)}
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'feat_key': feat_key,
                        'd_in': int(Xtr_f.shape[1]), 'feature_names': names,
                        'best_epoch': best['epoch'], 'dev_fusion_ndcg20': best['score'],
                        'loss': 'soft_label_cross_entropy_zero_hit_weight_0.25'},
                       sel / 'checkpoints' / f'{tag}.pth')
            results.append(rec)
            with torch.no_grad():
                w_hat = torch.softmax(
                    model(torch.tensor(Xdv_f, dtype=torch.float32)), dim=1).numpy()
            prev = family_best.get(feat_key)
            if prev is None or best['score'] > prev['score']:
                family_best[feat_key] = {'tag': tag, 'score': best['score'],
                                         'w1': w_hat[:, 0].copy()}
            print(f'    -> dev NDCG@20 = {best["score"]:.6f} '
                  f'(epoch {best["epoch"]}, {rec["train_seconds"]}s)', flush=True)

    # ---- is any RouterDev gain larger than noise?  (RouterDev only) ----
    dev_uniform_vec = per_user_ndcg(fused_rank(0.5, dev_cand), TOPK_EVAL)
    dev_uniform_score = float(dev_uniform_vec.mean())
    dev_significance = {'uniform_dev_score': dev_uniform_score}
    for feat_key, fb in family_best.items():
        vec = per_user_ndcg(fused_rank(fb['w1'], dev_cand), TOPK_EVAL)
        st = paired_bootstrap(vec, dev_uniform_vec, n_boot=5000, seed=BOOT_SEED)
        dev_significance[feat_key] = {'tag': fb['tag'], 'dev_score': float(vec.mean()),
                                      **st}
        print(f'  RouterDev {fb["tag"]} vs uniform: delta={st["delta"]:+.6f} '
              f'CI=[{st["ci_low"]:+.6f}, {st["ci_high"]:+.6f}] p={st["p_value"]:.4f}',
              flush=True)

    stat = [{'w_F1': float(wv), 'dev_fusion_ndcg20': dev_ndcg(np.full(len(dev_cand), wv))}
            for wv in STATIC_GRID]
    best_static = max(stat, key=lambda d: d['dev_fusion_ndcg20'])
    best_router = max(results, key=lambda r: r['dev_fusion_ndcg20'])

    with (sel / 'router_dev_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['router', 'feature_family', 'architecture', 'd_in', 'n_params',
                    'dev_fusion_ndcg20', 'best_epoch', 'epochs_run', 'checkpoint_sha256'])
        for r in results:
            w.writerow([r['tag'], r['feat_key'], r['kind'], r['d_in'], r['n_params'],
                        f'{r["dev_fusion_ndcg20"]:.6f}', r['best_epoch'], r['epochs_run'],
                        r['checkpoint_sha256']])
        w.writerow(['STATIC_w' + str(best_static['w_F1']), '-', 'constant', '-', 1,
                    f'{best_static["dev_fusion_ndcg20"]:.6f}', '-', '-', '-'])
        w.writerow(['UNIFORM_w0.5', '-', 'constant', '-', 1,
                    f'{uniform_dev:.6f}', '-', '-', '-'])

    summary = {
        'split_seed': SPLIT_SEED, 'split_fractions': list(SPLIT_FRACTIONS),
        'router_train_n': int(Xtr_h.shape[0]), 'router_dev_n': int(Xdv_h.shape[0]),
        'router_holdout_n': int(sum((assign[ds] == 'RouterHoldout').sum()
                                    for ds in DATASET_ORDER)),
        'router_h_dim': int(Xtr_h.shape[1]), 'router_p_dim': int(Xtr_p.shape[1]),
        'feature_names_h': names_h, 'feature_names_p': names_p,
        'scaler_hash': scaler_hash, 'results': results,
        'static_grid': stat, 'best_static': best_static,
        'router_dev_significance_vs_uniform': dev_significance,
        'uniform_dev_fusion_ndcg20': uniform_dev, 'best_router': best_router,
        'training': {'optimizer': 'AdamW', 'lr': a.lr, 'batch': a.batch,
                     'max_epochs': a.max_epochs, 'patience': a.patience, 'seed': a.seed,
                     'loss': 'soft-label CE, zero-hit sample weight 0.25',
                     'early_stop_monitor': 'RouterDev soft-fusion NDCG@20'},
        'test_read': False, 'git_sha': git_sha(),
    }
    (sel / 'router_dev_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'best router: {best_router["tag"]} dev={best_router["dev_fusion_ndcg20"]:.6f}',
          flush=True)
    print(f'best static: w_F1={best_static["w_F1"]} '
          f'dev={best_static["dev_fusion_ndcg20"]:.6f}', flush=True)
    print(f'uniform dev={uniform_dev:.6f}', flush=True)
    print('done (test untouched)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
