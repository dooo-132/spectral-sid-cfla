#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART B-E — train the disagreement-aware candidate gate.

Score definition (frozen, prompt section 19).  The gate only chooses how far a candidate
moves from Uniform toward the frozen LSE score:

    S(u,i) = (1 - g(u,i)) * Z_U(u,i) + g(u,i) * Z_L(u,i)

so `g = 0` reproduces Uniform exactly and `g = 1` reproduces LSE exactly.  `Z_U` and `Z_L`
are the *same* per-user candidate-union z-scores the previous round used (mean/std over the
union, std < 1e-8 -> 1), so the gate cannot redefine the scoring space.

Objective (prompt sections 26-31).  User-level listwise loss over users whose validation
target lies inside their frozen candidate union:

    L = -log softmax_i S(u,i)[i*]  + lambda_g * mean_i g(u,i)

Users whose target is outside the union carry weight 0 and are recorded as
`TARGET_OUTSIDE_UNION`.  The target's Only-F1/Only-F2/Both group is never an input and
never supervision.

Data layout: candidates are padded to a fixed per-user width with a boolean mask, so a
batch of 64 users is one tensor operation rather than 64 Python iterations.

Usage
  python3 dg_train_gate.py --outdir outputs/disagreement_gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dg_features import DATASET_ONEHOT, DATASETS, FEATURES, SHORT  # noqa: E402

LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2]
CONST_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_USERS = 64
LR = 1e-3
WD = 1e-4
SEED = 2025
ROLES = ('RouterTrain', 'RouterDev', 'RouterHoldout')


class LinearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class MLPGate(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------------------
# padded, vectorised view of one dataset+split
# ---------------------------------------------------------------------------
class Padded:
    """Padded candidate view so a user batch is a dense tensor operation."""

    def __init__(self, d, tag=''):
        self.dataset_tag = tag
        self.users = d['users']
        self.role_rows = d['role']                 # one role per candidate row
        self.starts = d['starts'].astype(np.int64)
        self.ends = d['ends'].astype(np.int64)
        n = len(self.users)
        self.n = n
        # role is constant within a user's block, so the per-user role is the block's role
        self.role = np.array([d['role'][int(self.starts[u])] for u in range(n)], dtype=object)
        widths = (self.ends - self.starts)
        W = int(widths.max())
        self.W = W
        X = np.zeros((n, W, d['Xz'].shape[1]), dtype=np.float32)
        # raw (unscaled) view, kept for diagnostics that must bucket on real units
        Xraw = np.zeros((n, W, d['X'].shape[1]), dtype=np.float64)
        mask = np.zeros((n, W), dtype=bool)
        rows = np.zeros((n, W), dtype=np.int64)          # index into the flat arrays
        for u in range(n):
            s, e = int(self.starts[u]), int(self.ends[u])
            w = e - s
            X[u, :w] = d['Xz'][s:e]
            Xraw[u, :w] = d['X'][s:e]
            mask[u, :w] = True
            rows[u, :w] = np.arange(s, e)
        self.X = X
        self.Xraw = Xraw
        self.mask = mask
        self.rows = rows
        # frozen score pieces, per user z-normalised exactly as last round
        # Frozen score pieces are already per-user z-normalised by the previous
        # round (Z_U, Z_L over the candidate union, std<1e-8 -> 1).  They are kept
        # FLAT so ranking/NDCG match the frozen evaluation exactly; only the
        # gate's feature tensor is padded.
        self.ZU_flat = np.asarray(d['U'], dtype=np.float64)
        self.ZL_flat = np.asarray(d['L'], dtype=np.float64)
        self.cand_item = np.asarray(d['cand_item'])
        # frozen per-factorization ranks of each candidate (1-based; 0/absent otherwise)
        self.crank1 = (d.get('cand_rank1') if 'cand_rank1' in d
                       else np.zeros(len(self.ZU_flat), dtype=np.int64))
        self.crank2 = (d.get('cand_rank2') if 'cand_rank2' in d
                       else np.zeros(len(self.ZU_flat), dtype=np.int64))
        # target position inside the padded row, or -1
        self.rel = np.full(n, -1, dtype=np.int64)
        for u in range(n):
            t = int(d['z_target'][u])
            if t < 0:
                continue
            s, e = int(self.starts[u]), int(self.ends[u])
            hit = np.where(d['cand_item'][s:e] == t)[0]
            if len(hit):
                self.rel[u] = int(hit[0])
        self.eligible = self.rel >= 0


def load_padded(outdir, split):
    out = {}
    for ds in DATASETS:
        p = Path(outdir) / 'features' / f'{ds}_{split}_gate_features.npz'
        if not p.exists():
            raise FileNotFoundError(p)
        out[ds] = dict(np.load(p, allow_pickle=True))
    return out


def fit_feature_scaler(cells):
    """Per-dataset mean/std over RouterTrain rows, continuous columns only."""
    cont = [i for i, f in enumerate(FEATURES) if f not in DATASET_ONEHOT]
    stats = {}
    for ds, d in cells.items():
        m = d['role'] == 'RouterTrain'
        X = d['X'][m][:, cont]
        mu = X.mean(axis=0); sd = X.std(axis=0)
        stats[ds] = {'mean': mu.tolist(), 'std': np.where(sd < 1e-8, 1.0, sd).tolist(),
                     'cont_idx': cont, 'n_fit': int(m.sum()),
                     'fitted_on': 'RouterTrain rows of this dataset'}
    return stats


def apply_scaler(cells, stats):
    for ds, d in cells.items():
        s = stats[ds]; cont = s['cont_idx']
        X = d['X'].astype(np.float64).copy()
        X[:, cont] = (X[:, cont] - np.asarray(s['mean'])) / np.asarray(s['std'])
        d['Xz'] = X.astype(np.float32)
    return cells


# ---------------------------------------------------------------------------
# vectorised scoring / ranking
# ---------------------------------------------------------------------------
@torch.no_grad()
def gate_all(model, P, batch=256):
    model.eval()
    out = np.empty((P.n, P.W), dtype=np.float32)
    X = torch.tensor(P.X)
    for s in range(0, P.n, batch):
        e = min(s + batch, P.n)
        out[s:e] = model(X[s:e]).numpy()
    return out


def ndcg_at_20(ranks):
    r = np.asarray(ranks)
    hit = (r >= 1) & (r <= 20)
    v = np.zeros(len(r)); v[hit] = 1.0 / np.log2(r[hit] + 1)
    return v


def flat_from_padded(P, G_pad):
    """Scatter a per-user padded gate tensor back to the flat candidate order."""
    out = np.empty(len(P.ZU_flat), dtype=np.float64)
    for u in range(P.n):
        a, b = int(P.starts[u]), int(P.ends[u])
        out[a:b] = G_pad[u, :b - a]
    return out


def flat_constant(P, gval):
    return np.full(len(P.ZU_flat), float(gval), dtype=np.float64)


def ranks_for(P, G_flat):
    """Target rank under S = (1-g) Z_U + g Z_L; -1 when the target is outside the union.

    Scoring happens in the FLAT candidate space, exactly as the frozen round did, so the
    gate cannot alter the normalisation or the ranking convention.
    """
    G_flat = np.asarray(G_flat, dtype=np.float64)
    S = (1.0 - G_flat) * P.ZU_flat + G_flat * P.ZL_flat
    ranks = np.full(P.n, -1, dtype=np.int64)
    for u in np.where(P.eligible)[0]:
        a = int(P.starts[u]) + int(P.rel[u])
        st = S[a]
        b = int(P.ends[u])
        ranks[u] = 1 + int((S[int(P.starts[u]):b] > st).sum())
    return ranks, S


def dev_objective(model, P):
    G_pad = gate_all(model, P)
    ranks, _ = ranks_for(P, flat_from_padded(P, G_pad))
    m = P.role == 'RouterDev'
    return float(ndcg_at_20(ranks[m]).mean()), G_pad


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _gather_flat(P, users, attr):
    """(B,W) padded gather of a flat per-candidate array."""
    src = getattr(P, attr)
    out = np.zeros((len(users), P.W), dtype=np.float64)
    for j, u in enumerate(users):
        a, b = int(P.starts[u]), int(P.ends[u])
        out[j, :b - a] = src[a:b]
    return out


def train_one(kind, P_train, P_dev, lam_g, seed, max_epochs, patience, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = P_train.X.shape[2]
    model = LinearGate(d_in) if kind == 'linear' else MLPGate(d_in)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    Xtr = torch.tensor(P_train.X)
    Mtr = torch.tensor(P_train.mask)
    users = np.where(P_train.eligible & (P_train.role == 'RouterTrain'))[0]
    best = {'score': -np.inf, 'state': None, 'epoch': -1}
    bad, hist = 0, []
    for ep in range(max_epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(users))
        tot, nb = 0.0, 0
        for s in range(0, len(users), BATCH_USERS):
            b = users[perm[s:s + BATCH_USERS]]
            g = model(Xtr[b])                                     # (B,W) padded
            sc = (1 - g) * torch.tensor(_gather_flat(P_train, b, 'ZU_flat')) \
                + g * torch.tensor(_gather_flat(P_train, b, 'ZL_flat'))
            sc = sc.masked_fill(~Mtr[b], float('-inf'))
            logp = torch.log_softmax(sc, dim=1)
            tgt = torch.tensor(P_train.rel[b])
            rank_loss = -logp.gather(1, tgt[:, None]).squeeze(1).mean()
            reg = g.mean()
            loss = rank_loss + lam_g * reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        score, _ = dev_objective(model, P_dev)
        hist.append({'epoch': ep + 1, 'train_loss': tot / max(nb, 1), 'dev_ndcg20': score})
        if verbose and (ep + 1) % 10 == 0:
            print(f'      ep{ep+1:3d} loss={tot/max(nb,1):.5f} dev={score:.6f}', flush=True)
        if score > best['score'] + 1e-9:
            best = {'score': score, 'epoch': ep + 1,
                    'state': {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best['state'])
    return model, best, hist


def pooled_dev(model, P_by_ds):
    vals = []
    for ds, P in P_by_ds.items():
        ranks, _ = ranks_for(P, flat_from_padded(P, gate_all(model, P)))
        m = P.role == 'RouterDev'
        vals.append(ndcg_at_20(ranks[m]))
    return float(np.concatenate(vals).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    a = ap.parse_args(argv)
    out = Path(a.outdir)

    print('[1/3] loading validation features and fitting the feature scaler')
    valid_cells = load_padded(out, 'valid')
    feature_scaler = fit_feature_scaler(valid_cells)
    # Persist the scaler so test uses EXACTLY the RouterTrain statistics, never ones
    # refitted on test rows.
    json.dump({'fitted_on': 'RouterTrain rows per dataset',
               'features': FEATURES, 'continuous_idx': feature_scaler[DATASETS[0]]['cont_idx'],
               'per_dataset': {ds: {'mean': feature_scaler[ds]['mean'],
                                    'std': feature_scaler[ds]['std'],
                                    'n_fit': feature_scaler[ds]['n_fit']}
                               for ds in DATASETS}},
              open(out / 'features' / 'FEATURE_SCALER.json', 'w'), indent=2)
    cells = apply_scaler(valid_cells, feature_scaler)
    P = {ds: Padded(cells[ds], ds) for ds in DATASETS}
    json.dump({'seed': a.seed, 'lambda_grid': LAMBDA_GRID, 'const_grid': CONST_GRID,
               'features': FEATURES, 'lr': LR, 'weight_decay': WD,
               'batch_users': BATCH_USERS, 'max_epochs': a.max_epochs,
               'patience': a.patience, 'roles': list(ROLES),
               'loss': 'listwise -log softmax(S)[target] + lambda_g * mean(g)',
               'selection': 'pooled RouterDev NDCG@20, tie -> fewer parameters',
               'feature_scaler': {ds: {k: v for k, v in fit_feature_scaler(load_padded(out,'valid'))[ds].items()
                                       if k in ('mean','std','cont_idx','n_fit','fitted_on')}
                                  for ds in DATASETS}},
              open(out / 'training_config.json', 'w'), indent=2)
    for ds in DATASETS:
        p = P[ds]
        print(f'  {SHORT[ds]:12s} users={p.n:>6d} W={p.W:>3d} eligible '
              f'train/dev/holdout = {int((p.eligible & (p.role=="RouterTrain")).sum())}/'
              f'{int((p.eligible & (p.role=="RouterDev")).sum())}/'
              f'{int((p.eligible & (p.role=="RouterHoldout")).sum())}', flush=True)

    print('[2/3] baselines on RouterDev')
    dev_rows = []
    # g = 0 (Uniform) and g = 1 (LSE) references, plus the constant grid
    for tag, gval in [('Uniform(g=0)', 0.0), ('LSE(g=1)', 1.0)] + \
                     [(f'Const(g={c})', c) for c in CONST_GRID]:
        vals, per = [], {}
        for ds in DATASETS:
            p = P[ds]
            ranks, _ = ranks_for(p, flat_constant(p, gval))
            m = p.role == 'RouterDev'
            v = ndcg_at_20(ranks[m]); vals.append(v); per[ds] = float(v.mean())
        score = float(np.concatenate(vals).mean())
        dev_rows.append({'model': tag, 'kind': 'constant', 'g': gval, 'lambda_g': None,
                         'params': 0, 'dev_ndcg20': score,
                         **{f'dev_{SHORT[ds]}': per[ds] for ds in DATASETS}})
        print(f'  {tag:14s} pooled dev NDCG@20 = {score:.6f}', flush=True)

    print('[3/3] training gates (one GLOBAL gate pooled over the four datasets)')
    # Training sees the pooled RouterTrain rows only.  Early stopping must score
    # RouterDev ONLY: using the pooled set mixes in the training users and selects on a
    # diluted, non-comparable number (that mistake was made and caught here).
    Ptr = _concat([P[ds] for ds in DATASETS], role='RouterTrain')
    Pdv = _concat([P[ds] for ds in DATASETS], role='RouterDev')
    hold = sum(int((P[ds].eligible & (P[ds].role == 'RouterHoldout')).sum()) for ds in DATASETS)
    print(f'  pooled eligible users: RouterTrain={int(Ptr.eligible.sum())} '
          f'RouterDev={int(Pdv.eligible.sum())} RouterHoldout={hold}  W={Ptr.W}')

    trained = []
    for kind in ('linear', 'mlp'):
        for lam in LAMBDA_GRID:
            t0 = time.time()
            model, best, hist = train_one(kind, Ptr, Pdv, lam, a.seed, a.max_epochs,
                                          a.patience, verbose=True)
            tag = f'{kind}_lambda{lam:g}'
            pooled = pooled_dev(model, P)
            cfg_here = {'model': tag, 'features': FEATURES, 'lambda_g': lam,
                        'seed': a.seed, 'arch': kind,
                        'splits': ['RouterTrain', 'RouterDev', 'RouterHoldout'],
                        'lr': LR, 'weight_decay': WD, 'batch_users': BATCH_USERS,
                        'max_epochs': a.max_epochs, 'patience': a.patience}
            from cc_common import config_hash as _ch
            run_dir = out / 'runs' / f'{tag}_{a.seed}_{_ch(cfg_here)[:8]}'
            if run_dir.exists():
                raise SystemExit(f'HARD FAIL: {run_dir} already exists')
            run_dir.mkdir(parents=True)
            (run_dir / 'config.json').write_text(json.dumps(cfg_here, indent=2))
            ck = run_dir / 'gate.pt'
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'lambda_g': lam,
                        'features': FEATURES, 'd_in': int(Ptr.X.shape[2]),
                        'best_epoch': best['epoch'], 'dev_pooled_ndcg20': pooled,
                        'seed': a.seed}, ck)
            row = {'model': tag, 'kind': kind, 'g': None, 'lambda_g': lam,
                   'params': count_params(model), 'dev_ndcg20': pooled,
                   'best_epoch': best['epoch'], 'epochs_run': len(hist),
                   'seconds': round(time.time() - t0, 1)}
            for ds in DATASETS:
                p = P[ds]
                ranks, _ = ranks_for(p, flat_from_padded(p, gate_all(model, p)))
                m = p.role == 'RouterDev'
                row[f'dev_{SHORT[ds]}'] = float(ndcg_at_20(ranks[m]).mean())
            dev_rows.append(row)
            trained.append((row, model))
            print(f'  {tag:22s} params={row["params"]:>5d} '
                  f'pooled dev NDCG@20 = {pooled:.6f} '
                  f'(best epoch {best["epoch"]}, {row["seconds"]}s)', flush=True)

    # ---- selection: pooled RouterDev NDCG@20, tie -> fewer parameters ----
    dev_rows.sort(key=lambda r: (-r['dev_ndcg20'], r['params']))
    winner = dev_rows[0]
    print(f'  WINNER: {winner["model"]}  pooled dev NDCG@20={winner["dev_ndcg20"]:.6f} '
          f'params={winner["params"]}')

    for r in dev_rows:
        r.setdefault('best_epoch', None); r.setdefault('epochs_run', None)
        r.setdefault('seconds', None)
    import csv as _csv
    keys = ['model', 'kind', 'g', 'lambda_g', 'params', 'dev_ndcg20'] + \
           [f'dev_{SHORT[ds]}' for ds in DATASETS] + ['best_epoch', 'epochs_run', 'seconds']
    with (out / 'router_dev_results.csv').open('w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in dev_rows:
            w.writerow({k: r.get(k) for k in keys})

    json.dump({'eligible_users': {
                   'RouterTrain': int(Ptr.eligible.sum()),
                   'RouterDev': int(Pdv.eligible.sum()),
                   'RouterHoldout': int(hold)},
               'total_users_per_role': {
                   r: int(sum((P[ds].role == r).sum() for ds in DATASETS)) for r in ROLES}},
              open(out / 'eligible_users.json', 'w'), indent=2)
    json.dump({'winner': winner['model'], 'selection_rule':
               'pooled RouterDev NDCG@20; tie -> fewer parameters',
               'feature_scaler_file': str(out / 'features' / 'FEATURE_SCALER.json')},
              open(out / 'selected_gate.json', 'w'), indent=2)
    print(f'\nwrote {out/"router_dev_results.csv"} and selected_gate.json')
    return 0


def _concat(Ps, role):
    """One pooled Padded view containing ONLY the users of `role` from each dataset.

    Filtering by role is the whole point: training and early stopping must see disjoint
    user sets, otherwise the dev number is diluted by training rows and is not comparable
    to the frozen baselines.
    """
    sel = []
    for p in Ps:
        m = (p.role == role)
        if not m.any():
            continue
        q = Padded.__new__(Padded)
        q.dataset_tag = f'{p.dataset_tag}_{role}'
        idx = np.where(m)[0]
        q.X = p.X[idx]
        q.Xraw = p.Xraw[idx]
        q.mask = p.mask[idx]
        q.W = p.W
        q.n = len(idx)
        q.users = p.users[idx]
        q.role = p.role[idx]
        q.rel = p.rel[idx]
        q.eligible = p.eligible[idx]
        q.starts = p.starts[idx]
        q.ends = p.ends[idx]
        # flat score arrays must be sliced by each user's candidate block
        keep = np.concatenate([np.arange(int(p.starts[u]), int(p.ends[u])) for u in idx]) \
            if len(idx) else np.array([], dtype=np.int64)
        q.ZU_flat = p.ZU_flat[keep]
        q.ZL_flat = p.ZL_flat[keep]
        q.cand_item = p.cand_item[keep]
        # re-base starts/ends onto the sliced flat arrays
        widths = q.ends - q.starts
        q.starts = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
        q.ends = (q.starts + widths).astype(np.int64)
        q.rows = np.zeros_like(q.mask, dtype=np.int64)
        sel.append(q)
    if not sel:
        raise ValueError(f'no users with role {role}')

    W = max(q.W for q in sel)
    out = Padded.__new__(Padded)
    out.dataset_tag = 'pooled_' + role
    out.W = W

    def repad(a, fill):
        if a.shape[1] == W:
            return a
        tail = np.full((a.shape[0], W - a.shape[1]) + a.shape[2:], fill, dtype=a.dtype)
        return np.concatenate([a, tail], axis=1)

    out.X = np.concatenate([repad(q.X, 0.0) for q in sel], axis=0)
    out.Xraw = np.concatenate([repad(q.Xraw, 0.0) for q in sel], axis=0)
    out.mask = np.concatenate([repad(q.mask, False) for q in sel], axis=0)
    out.ZU_flat = np.concatenate([q.ZU_flat for q in sel])
    out.ZL_flat = np.concatenate([q.ZL_flat for q in sel])
    out.cand_item = np.concatenate([q.cand_item for q in sel])
    out.rel = np.concatenate([q.rel for q in sel])
    out.eligible = np.concatenate([q.eligible for q in sel])
    out.role = np.concatenate([q.role for q in sel])
    out.users = np.concatenate([q.users for q in sel])
    # re-base starts/ends onto the concatenated flat arrays
    widths = np.concatenate([q.ends - q.starts for q in sel])
    out.starts = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    out.ends = (out.starts + widths).astype(np.int64)
    out.rows = np.zeros_like(out.mask, dtype=np.int64)
    out.n = out.X.shape[0]
    assert len(out.ZU_flat) == int(widths.sum()), 'flat array length mismatch'
    return out


if __name__ == '__main__':
    sys.exit(main())
