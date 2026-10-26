#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-factorization candidate-level fusion — stage driver.

    stage=baselines   (F)   validation-only lambda selection + baseline scores
    stage=gate        (G,H) train Gate-S4 / Gate-M5 on VALID router users only
    stage=freeze      (I)   MULTIFACTOR_FUSION_FREEZE_MANIFEST.json
    stage=test        (J,K) frozen test evaluation + paired bootstrap
    stage=coverage    (L,M) coverage, RQ-only value, unique-factorization groups
    stage=efficiency  (M)   compute / VRAM accounting

Selection discipline: every hyper-parameter (lambda) and the gate checkpoint are
chosen on the canonical VALIDATION split among RouterTrain / RouterDev users.  The
test split is read only by stage=test, after the freeze manifest exists.

Frozen validation scalers live in `validation/validation_nll_scalers.json`; the
frozen lambda and gate checkpoints are recorded in the freeze manifest and are
re-loaded, never re-selected, on the test path.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (DATASETS, FLABEL, M5, OUT, S4, SHORT, config_hash,  # noqa: E402
                       load_beam_cell, read_json, sha256_file, write_json)
from mf_core import (TOPK, fit_scalers, group_view, load_split,  # noqa: E402
                     per_user_lse_rows, rank_and_metrics, target_rank_from_scores,
                     user_zscore, z_of)
from mf_gate import D_IN, build_features, feature_names, n_params, torch_module  # noqa: E402

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
METRIC_KEYS = ('Recall@5', 'Recall@10', 'Recall@20',
               'NDCG@5', 'NDCG@10', 'NDCG@20')
SPLIT_CSV = Path('outputs/disagreement_gate/splits/router_user_split.csv')
HERE_DEV = Path(__file__).resolve().parent

# the frozen reference reproduced in phase B (bit-exact against its own artifacts)
DYNAMIC_REF = {
    'Beauty': 0.046237789043916025,
    'Cell_Phones_and_Accessories': 0.108456,
    'Health_and_Personal_Care': 0.095827,
    'Toys_and_Games': 0.042212,
}


def pool_factorizations(pool):
    return S4 if pool == 'spectral4' else M5


def load_targets(ds, split='valid'):
    """Canonical 1-based target item id per user, from the beam dumps.

    All five dumps carry the same target array (verified in the artifact matrix),
    so one read is a cross-check of the other four.
    """
    t = load_beam_cell('spectral_k2', ds, split)['target']
    for f in ('spectral_k3', 'spectral_k4', 'spectral_k5', 'rqvae3'):
        if not np.array_equal(load_beam_cell(f, ds, split)['target'], t):
            raise AssertionError(f'{ds}/{split}: target mismatch in {f}')
    return t + 1


def load_scalers():
    p = OUT / 'validation' / 'validation_nll_scalers.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} missing: run stage=baselines first')
    j = read_json(p)
    return {ds: {f: tuple(v) for f, v in j['datasets'][ds]['scalers'].items()}
            for ds in j['datasets']}


def score_block(df, f_list):
    """Grouped per-candidate arrays: Z (zNLL), P (probability), present, rank."""
    Z = z_of(df, f_list)
    P = np.stack([df[f'prob_{FLABEL[f]}'].values.astype(np.float64) for f in f_list], 1)
    present = np.stack([df[f'present_{FLABEL[f]}'].values.astype(bool) for f in f_list], 1)
    R = np.stack([np.where(df[f'present_{FLABEL[f]}'].values,
                           df[f'rank_{FLABEL[f]}'].values, 0) for f in f_list], 1)
    return Z, P, present, R


def gate_predict(gate, X, device='cuda', batch=32768):
    import torch
    out = np.empty((X.shape[0], X.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            j = min(i + batch, X.shape[0])
            t = torch.as_tensor(X[i:j], dtype=torch.float32, device=device)
            out[i:j] = gate(t).cpu().numpy()
    return out


def load_gate(pool, device='cuda'):
    """Rebuild the frozen gate from its checkpoint (state + scaler + arch)."""
    import torch
    info = read_json(OUT / 'validation' / 'gate_training.json')[pool]
    # `weights_only=False`: the checkpoint carries plain metadata alongside the state
    # dict (feature scaler mean/std as numpy arrays, feature names, training history).
    # It was written by this round's own training stage, so the source is trusted.
    ck = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
    Gate = torch_module()
    model = Gate(ck['d_in'], ck['n_f']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck, info


# ---------------------------------------------------------------------------
# stage F — baselines on VALIDATION
# ---------------------------------------------------------------------------
def stage_baselines(args):
    import pandas as pd
    (OUT / 'validation').mkdir(parents=True, exist_ok=True)
    out = {'stage': 'baselines', 'selection_split': 'canonical VALIDATION',
           'pools': {}, 'lambda_grid': list(LAMBDA_GRID),
           'target_source': 'beam-dump target_item_ids (0-based row) + 1'}
    scal_out = {'fitted_on': 'ALL canonical validation candidates of that '
                             '(dataset, factorization); never targets-only, never test',
                'split': 'valid', 'datasets': {}}

    for pool in ('spectral4', 'multi5'):
        f_list = pool_factorizations(pool)
        out['pools'][pool] = {}
        for ds in DATASETS:
            df, upath = load_split(OUT, ds, pool, 'valid')
            target = load_targets(ds, 'valid')
            scalers = fit_scalers(df, f_list)
            df.attrs['scalers'] = scalers
            Z, P, present, R = score_block(df, f_list)
            users, starts, ends, order = group_view(df.user.values,
                                                    df.candidate_item.values)
            items_g = df.candidate_item.values[order]
            Zo, Po = Z[order], P[order]
            # validation target per grouped user, and its row position in the block
            tgt_g = np.array([target[u] for u in range(len(users))], dtype=np.int64)
            tgt_pos = np.full(len(users), -1, dtype=np.int64)
            for u, (s, e) in enumerate(zip(starts, ends)):
                w = np.where(items_g[s:e] == tgt_g[u])[0]
                if len(w):
                    tgt_pos[u] = s + int(w[0])
            uni = Po.mean(axis=1)
            lse = per_user_lse_rows(-Zo)
            res = {}
            for name, sc in (('Uniform', uni), ('LSE', lse)):
                res[name] = rank_and_metrics(sc, items_g, tgt_g, starts, ends)['metrics']
            uni_z = user_zscore(uni, starts, ends)
            lse_z = user_zscore(lse, starts, ends)
            lam_table, lam_best = {}, None
            for lam in LAMBDA_GRID:
                h = lam * uni_z + (1 - lam) * lse_z
                m = rank_and_metrics(h, items_g, tgt_g, starts, ends)['metrics']
                lam_table[str(lam)] = m
                if lam_best is None or m['NDCG@20'] > lam_table[str(lam_best)]['NDCG@20']:
                    lam_best = lam
            # likelihood-only diagnostic (secondary formula)
            lik = (-Zo).mean(axis=1)
            res['MeanNegZNLL'] = rank_and_metrics(lik, items_g, tgt_g,
                                                  starts, ends)['metrics']
            res['Oracle'] = rank_and_metrics(
                np.where(np.arange(len(items_g)) == tgt_pos.repeat(ends - starts), 1.0, 0.0),
                items_g, tgt_g, starts, ends)['metrics']
            out['pools'][pool][ds] = {
                'scalers': {f: list(v) for f, v in scalers.items()},
                'methods': res, 'lambda_table': lam_table, 'lambda_selected': lam_best,
                'union_parquet': str(upath), 'union_sha256': sha256_file(upath),
                'n_users': int(len(users)), 'n_pairs': int(len(df)),
                'target_in_union': int((tgt_pos >= 0).sum())}
            scal_out['datasets'].setdefault(ds, {'scalers': {}})
            scal_out['datasets'][ds]['scalers'].update(
                {f: list(v) for f, v in scalers.items()})
            print(f'  [{pool}] {SHORT[ds]:12s} valid NDCG@20: Uniform='
                  f'{res["Uniform"]["NDCG@20"]:.6f} LSE={res["LSE"]["NDCG@20"]:.6f} '
                  f'lam*={lam_best} Oracle={res["Oracle"]["NDCG@20"]:.6f}', flush=True)

    write_json(OUT / 'validation' / 'validation_nll_scalers.json', scal_out)
    write_json(OUT / 'validation' / 'validation_baseline_selection.json', out)
    rows = []
    for pool in ('spectral4', 'multi5'):
        for ds in DATASETS:
            d = out['pools'][pool][ds]
            for name, m in d['methods'].items():
                rows.append({'dataset': ds, 'pool': pool, 'method': name, **m})
            for lam, m in d['lambda_table'].items():
                rows.append({'dataset': ds, 'pool': pool,
                             'method': f'Hybrid_lam{lam}', **m})
    pd.DataFrame(rows).to_csv(OUT / 'validation' / 'validation_results.csv', index=False)
    print(f'wrote {OUT / "validation" / "validation_results.csv"}')
    return out


# ---------------------------------------------------------------------------
# stage G+H — gate training
# ---------------------------------------------------------------------------
def load_user_split():
    """Reuse the historical RouterTrain / RouterDev / RouterHoldout split.

    Prompt section 40 says to reuse it if it applies.  It does: the file assigns
    every user of all four datasets to exactly one role, independent of the candidate
    pool, so nothing about it needs re-deriving.  The source file is read-only and is
    not modified; only its role assignment is consumed.
    """
    import pandas as pd
    d = pd.read_csv(SPLIT_CSV)
    return {ds: d[d.dataset == ds][['user', 'split']] for ds in d.dataset.unique()}


def build_role_blocks(pool, role, scalers_all, split_map, d_in):
    """Feature/probability/zNLL blocks for one router role, pooled over datasets."""
    import pandas as pd
    f_list = pool_factorizations(pool)
    Xs, Ps, Zs, Ts, Ds, bounds, user_lens = [], [], [], [], [], [], []
    for ds in DATASETS:
        df, _ = load_split(OUT, ds, pool, 'valid')
        df.attrs['scalers'] = scalers_all[ds]
        target = load_targets(ds, 'valid')
        Z, P, present, R = score_block(df, f_list)
        users, starts, ends, order = group_view(df.user.values, df.candidate_item.values)
        role_of = split_map[ds].set_index('user')['split'].reindex(users).values
        sel = np.where(role_of == role)[0]
        if len(sel) == 0:
            continue
        X = build_features(df, f_list, Z, P, present, R)[order]
        user_lens.append((ends - starts)[sel].astype(np.int64))
        idx = np.concatenate([np.arange(starts[u], ends[u]) for u in sel])
        Xs.append(X[idx]); Ps.append(P[order][idx]); Zs.append(Z[order][idx])
        # `users` are canonical 1-based ids; `target` is a 0-based row array
        Tg = target[users[sel] - 1]
        Ts.append(Tg)
        # target row position inside the concatenated block
        tpos = np.full(len(sel), -1, dtype=np.int64)
        items_g = df.candidate_item.values[order]
        off = 0
        for k, u in enumerate(sel):
            seg = items_g[starts[u]:ends[u]]
            w = np.where(seg == Tg[k])[0]
            if len(w):
                tpos[k] = off + int(w[0])
            off += ends[u] - starts[u]
        bounds.append((off, tpos))
        Ds.append(np.full(len(sel), ds, dtype=object))
    # PER-USER group bounds over the concatenated block.  `bounds` holds one entry
    # per dataset: (n_rows_of_that_dataset, per-user target row positions).  The
    # per-user candidate counts are derivable from the per-dataset slice lengths and
    # the row counts, so they are reconstructed here rather than assumed.
    cell_rows = np.array([b[0] for b in bounds], dtype=np.int64)
    block_tpos = [b[1] for b in bounds]
    n_users_per_cell = np.array([len(t) for t in block_tpos], dtype=np.int64)
    cell_starts = np.concatenate([[0], np.cumsum(cell_rows)[:-1]])
    user_len = np.concatenate([len_per_ds for len_per_ds in user_lens])
    ends = np.cumsum(user_len).astype(np.int64)
    starts = ends - user_len
    # absolute row position of each user's target inside the concatenated block
    tpos = np.concatenate(block_tpos)
    offsets = np.repeat(cell_starts, n_users_per_cell)
    tpos = np.where(tpos >= 0, tpos + offsets, -1)
    target = np.concatenate(Ts)
    n_rows = int(cell_rows.sum())
    if len(target) != len(user_len):
        raise AssertionError(f'user count {len(target)} != group lengths {len(user_len)}')
    if int(ends[-1]) != n_rows:
        raise AssertionError(f'block rows {int(ends[-1])} != feature rows {n_rows}')
    if np.any(ends <= starts):
        raise AssertionError('empty user block encountered')
    return {'X': np.concatenate(Xs), 'P': np.concatenate(Ps), 'Z': np.concatenate(Zs),
            'target': target, 'dataset': np.concatenate(Ds),
            'starts': starts, 'ends': ends, 'target_pos': tpos}


def stage_gate(args):
    import torch
    split_map = load_user_split()
    scalers_all = load_scalers()
    Gate = torch_module()
    results = {}

    for pool in ('spectral4', 'multi5'):
        f_list = pool_factorizations(pool)
        blocks = {role: build_role_blocks(pool, role, scalers_all, split_map, D_IN)
                  for role in ('RouterTrain', 'RouterDev', 'RouterHoldout')}
        for role, b in blocks.items():
            print(f'  [{pool}] {role:15s} users={len(b["target"])} '
                  f'candidates={len(b["X"])} target_in_union={int((b["target_pos"]>=0).sum())}',
                  flush=True)
        mu = blocks['RouterTrain']['X'].reshape(-1, D_IN).mean(axis=0)
        sd = blocks['RouterTrain']['X'].reshape(-1, D_IN).std(axis=0)
        sd[sd < 1e-9] = 1.0
        for b in blocks.values():
            b['Xn'] = ((b['X'] - mu) / sd).astype(np.float32)

        def infer(block, model):
            W = gate_predict(model, block['Xn'], device=args.device)
            return (W * block['P']).sum(axis=1)

        def block_rank(scores, block):
            """1-based target rank per user, from the precomputed target row positions."""
            return target_rank_from_scores(scores, block['target_pos'],
                                           block['starts'], block['ends'])

        def dev_ndcg(block, model):
            s = infer(block, model)
            return rank_and_metrics(s, target_rank=block_rank(s, block),
                                    starts=block['starts'], ends=block['ends'],
                                    topk=(20,))['metrics']['NDCG@20']

        torch.manual_seed(2025)
        rng = np.random.default_rng(2025)
        model = Gate(D_IN, len(f_list)).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr = blocks['RouterTrain']
        n_tr = len(tr['target'])

        # Precompute per-user row ranges ONCE (start offsets, then lengths).  The
        # inner training loop then only concatenates precomputed slices, instead of
        # building a boolean mask over every training row for every minibatch of
        # every epoch.  `starts` has one entry per dataset, so the per-user ranges
        # are rebuilt from the per-user candidate counts.
        tr_rows = [np.arange(s, e) for s, e in zip(tr['starts'], tr['ends'])]
        if len(tr_rows) != n_tr:
            raise AssertionError(f'row ranges {len(tr_rows)} != users {n_tr}')
        Xt = torch.as_tensor(tr['Xn'], dtype=torch.float32, device=args.device)
        Pt = torch.as_tensor(tr['P'], dtype=torch.float32, device=args.device)
        tp = torch.as_tensor(tr['target_pos'], dtype=torch.long, device=args.device)
        best = {'ndcg': -1.0, 'epoch': -1, 'state': None}
        hist = []
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            perm = rng.permutation(n_tr)
            tot, nb = 0.0, 0
            for i in range(0, n_tr, args.batch_users):
                uidx = perm[i:i + args.batch_users]
                rows_np = np.concatenate([tr_rows[u] for u in uidx])
                rows = torch.as_tensor(rows_np, dtype=torch.long, device=args.device)
                W = model(Xt[rows])
                s = (W * Pt[rows]).sum(dim=1)
                # group boundaries inside this minibatch
                lens = np.array([len(tr_rows[u]) for u in uidx], dtype=np.int64)
                bnd = np.concatenate([[0], np.cumsum(lens)])
                inv = torch.as_tensor(np.repeat(np.arange(len(uidx)), lens),
                                      dtype=torch.long, device=args.device)
                m = torch.full((len(uidx),), -1e30, device=args.device)
                m = m.scatter_reduce(0, inv, s, reduce='amax', include_self=True)
                ex = torch.exp(s - m[inv])
                den = torch.zeros_like(m).scatter_add_(0, inv, ex)
                logp = (s - m[inv]) - torch.log(den[inv])
                # absolute target row -> minibatch-local row
                tpos_abs = tp[uidx].detach().cpu().numpy()
                valid = np.where(tpos_abs >= 0)[0]
                if len(valid) == 0:
                    continue
                row_of = {int(r): j for j, r in enumerate(rows_np)}
                keep = np.array([row_of[int(tpos_abs[u])] for u in valid],
                                dtype=np.int64)
                loss = -logp[torch.as_tensor(keep, dtype=torch.long,
                                             device=args.device)].mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            dev = dev_ndcg(blocks['RouterDev'], model)
            hist.append({'epoch': epoch, 'train_loss': tot / max(nb, 1),
                         'RouterDev_ndcg20': dev})
            if dev > best['ndcg'] + 1e-9:
                best = {'ndcg': dev, 'epoch': epoch,
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
            if epoch - best['epoch'] >= args.patience:
                break
        model.load_state_dict(best['state'])
        model.eval()

        ckdir = OUT / 'runs' / f'gate_{pool}'
        ckdir.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'd_in': D_IN,
                    'n_f': len(f_list), 'feat_mu': mu, 'feat_sd': sd,
                    'feature_names': feature_names()[0],
                    'pool': pool, 'factorizations': list(f_list),
                    'best_epoch': best['epoch'], 'best_dev_ndcg20': best['ndcg'],
                    'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4,
                    'max_epochs': args.max_epochs, 'patience': args.patience,
                    'batch_users': args.batch_users, 'seed': 2025,
                    'feature_scaler': 'mean/std fitted on pooled RouterTrain rows',
                    'training_split': 'canonical VALIDATION, RouterTrain users only',
                    'selection_metric': 'pooled RouterDev NDCG@20'},
                   ckdir / 'gate.pt')
        write_json(ckdir / 'config.json',
                   {'pool': pool, 'factorizations': list(f_list), 'd_in': D_IN,
                    'n_f': len(f_list), 'arch': 'Linear(d,64)-GELU-Dropout(0.1)-'
                                                'Linear(64,32)-GELU-Dropout(0.1)-Linear(32,1)'
                                                ' + shared softmax over F',
                    'lr': 1e-3, 'weight_decay': 1e-4, 'batch_users': args.batch_users,
                    'max_epochs': args.max_epochs, 'patience': args.patience,
                    'seed': 2025, 'n_params': n_params(model)})
        write_json(ckdir / 'train_log.json', hist)
        entry = {'checkpoint': str(ckdir / 'gate.pt'),
                 'checkpoint_sha256': sha256_file(ckdir / 'gate.pt'),
                 'n_params': n_params(model), 'd_in': D_IN, 'n_f': len(f_list),
                 'best_epoch': best['epoch'], 'best_dev_ndcg20': best['ndcg'],
                 'epochs_run': len(hist),
                 'feature_names': feature_names()[0],
                 'feature_dim_note': feature_names()[1]}
        for role in blocks:
            s = infer(blocks[role], model)
            m = rank_and_metrics(s, target_rank=block_rank(s, blocks[role]),
                                 starts=blocks[role]['starts'],
                                 ends=blocks[role]['ends'], topk=TOPK)['metrics']
            entry[f'{role}_gate_ndcg20'] = m['NDCG@20']
            entry[f'{role}_gate_recall20'] = m['Recall@20']
            print(f'  [{pool}] {role:15s} Gate NDCG@20={m["NDCG@20"]:.6f}', flush=True)
        results[pool] = entry
        del Xt, Pt, tp
        torch.cuda.empty_cache()
    write_json(OUT / 'validation' / 'gate_training.json', results)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['baselines', 'gate', 'freeze', 'test', 'coverage',
                             'efficiency'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-users', type=int, default=128)
    ap.add_argument('--bootstrap', type=int, default=5000)
    a = ap.parse_args(argv)
    if a.stage == 'baselines':
        stage_baselines(a)
    elif a.stage == 'gate':
        stage_gate(a)
    elif a.stage == 'freeze':
        stage_freeze(a)
    elif a.stage == 'test':
        stage_test(a)
    elif a.stage == 'coverage':
        stage_coverage(a)
    elif a.stage == 'efficiency':
        stage_efficiency(a)
    else:
        raise SystemExit(f'stage {a.stage} not implemented in this revision')
    return 0


# ---------------------------------------------------------------------------
# stage I — freeze manifest
# ---------------------------------------------------------------------------
def _sha(p):
    return sha256_file(p)


def stage_freeze(args):
    import subprocess
    val = read_json(OUT / 'validation' / 'validation_baseline_selection.json')
    gates = read_json(OUT / 'validation' / 'gate_training.json')
    repro = read_json(OUT / 'DYNAMIC_FUSION_REPRODUCTION.json')
    matrix = OUT / 'FACTORISATION_ARTIFACT_MATRIX.csv'
    git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[3]),
                             capture_output=True, text=True).stdout.strip()
    git_branch = subprocess.run(['git', 'branch', '--show-current'],
                                cwd=str(Path(__file__).resolve().parents[3]),
                                capture_output=True, text=True).stdout.strip()

    art = {'git_sha': git_sha, 'git_branch': git_branch,
           'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'test_read_at_freeze_time': False,
           'preflight': read_json(OUT / 'OOM_PREFLIGHT.json'),
           'config_revision': read_json(OUT / '_CONFIG_REVISIONS.json'),
           'candidate_union_definition': {
               'PoolA_spectral4': 'C_K2 u C_K3 u C_K4 u C_K5',
               'PoolB_multi5': 'C_K2 u C_K3 u C_K4 u C_K5 u C_RQVAE',
               'beam_size_per_factorization': 30,
               'target_never_added': True,
               'sources': {'spectral_k2': 'independent historical dump',
                           'spectral_k4': 'independent historical dump',
                           'spectral_k5': 'independent historical dump',
                           'spectral_k3': 'frozen shared-model beam (control token)',
                           'rqvae3': 'frozen shared-model beam (control token)'}},
           'nll_scalers': read_json(OUT / 'validation' / 'validation_nll_scalers.json'),
           'scalers_are_validation_fitted': True,
           'selected_lambda': {pool: {ds: val['pools'][pool][ds]['lambda_selected']
                                      for ds in val['pools'][pool]}
                               for pool in val['pools']},
           'lambda_grid': val['lambda_grid'],
           'formulas': {
               'S4_uniform': 'mean_{F in K2,K3,K4,K5} P_F(i|u)',
               'S4_LSE': 'logsumexp(-zNLL_K2,-zNLL_K3,-zNLL_K4,-zNLL_K5)',
               'M5_uniform': 'mean_{F in K2..K5,RQ} P_F(i|u)',
               'M5_LSE': 'logsumexp(-zNLL_F, F in M5)',
               'hybrid': 'lam*Z(Uniform) + (1-lam)*Z(LSE), Z = per-user z-score',
               'gate': 'sum_F w_F(u,i) * P_F(u,i)',
               'gate_likelihood': 'sum_F w_F(u,i) * (-zNLL_F(u,i))',
               'zNLL': '(TotalNLL_F - mu_F)/sigma_F, mu/sigma fitted on ALL validation candidates'},
           'gate_architecture': 'Linear(d,64)-GELU-Dropout(0.1)-Linear(64,32)-GELU-'
                                'Dropout(0.1)-Linear(32,1) shared over F + softmax across F',
           'gate_training': {pool: {k: gates[pool][k] for k in
                                    ('n_params', 'd_in', 'n_f', 'best_epoch',
                                     'best_dev_ndcg20', 'epochs_run',
                                     'RouterTrain_gate_ndcg20',
                                     'RouterDev_gate_ndcg20',
                                     'RouterHoldout_gate_ndcg20')}
                             for pool in gates},
           'gate_checkpoints': {pool: {'path': gates[pool]['checkpoint'],
                                       'sha256': gates[pool]['checkpoint_sha256']}
                                for pool in gates},
           'validation_results': {pool: {ds: val['pools'][pool][ds]['methods']
                                         for ds in val['pools'][pool]}
                                  for pool in val['pools']},
           'validation_lambda_table': {pool: {ds: val['pools'][pool][ds]['lambda_table']
                                              for ds in val['pools'][pool]}
                                       for pool in val['pools']},
           'dynamic_fusion_reproduction': {
               'max_abs_error_vs_frozen': repro['max_abs_error_vs_frozen'],
               'reproduced': repro['reproduced'],
               'gamma_eta': {k: [v['gamma'], v['eta']] for k, v in repro['cells'].items()}},
           'artifacts': {
               'factorisation_matrix': {'path': str(matrix), 'sha256': _sha(matrix)},
               'valid_union_stats': {str(p): _sha(p) for p in
                                     sorted((OUT / 'validation').glob('valid_union_stats.csv'))},
               'gate_config_s4': _sha(OUT / 'runs' / 'gate_spectral4' / 'config.json'),
               'gate_config_m5': _sha(OUT / 'runs' / 'gate_multi5' / 'config.json')},
           'dynamic_reference_ndcg20': DYNAMIC_REF,
           'api': {'device': args.device},
           'note': 'Written before the test split is read. Nothing below may change '
                   'after this file exists (prompt section 46).'}
    write_json(OUT / 'MULTIFACTOR_FUSION_FREEZE_MANIFEST.json', art)
    print(f"freeze manifest written: git {git_sha[:8]} on {git_branch}")
    print(f"  dynamic reproduction max|err| = {repro['max_abs_error_vs_frozen']}")
    for pool in gates:
        print(f"  {pool}: {gates[pool]['n_params']} params, best epoch "
              f"{gates[pool]['best_epoch']}, RouterDev {gates[pool]['best_dev_ndcg20']:.6f}")
    return art


# ---------------------------------------------------------------------------
# stage J+K — frozen test evaluation + bootstrap
# ---------------------------------------------------------------------------
def _method_packs(split, device='cuda'):
    """packed[pool][ds][method] = rank_and_metrics output, using frozen settings."""
    from mf_eval import Cell, dynamic_scores_on_union
    val = read_json(OUT / 'validation' / 'validation_baseline_selection.json')
    scalers = load_scalers()
    gate_info = read_json(OUT / 'validation' / 'gate_training.json')
    packed, dyn = {}, {}
    for pool in ('spectral4', 'multi5'):
        packed[pool], dyn[pool] = {}, {}
        for ds in DATASETS:
            cell = Cell(ds, pool, split, scalers,
                        val['pools'][pool][ds]['lambda_selected'], device=device)
            packs, scores = cell.packs()
            # the frozen old DynamicFusion, replayed on this union, target-free scoring
            d = dynamic_scores_on_union(ds, split, cell)
            packs['OldDynamicFusion'] = rank_and_metrics(
                d['score'], cell.items, cell.target, cell.starts, cell.ends)
            packed[pool][ds] = packs
            dyn[pool][ds] = {k: v for k, v in d.items() if k != 'score'}
            print(f'  [{pool}] {SHORT[ds]:12s} test NDCG@20 Gate='
                  f'{packs["Gate"]["metrics"]["NDCG@20"]:.6f} '
                  f'OldDynamicFusion={packs["OldDynamicFusion"]["metrics"]["NDCG@20"]:.6f} '
                  f'Uniform={packs["Uniform"]["metrics"]["NDCG@20"]:.6f} '
                  f'LSE={packs["LSE"]["metrics"]["NDCG@20"]:.6f}', flush=True)
    return packed, dyn


def stage_test(args):
    import pandas as pd
    from mf_eval import paired_bootstrap
    freeze = read_json(OUT / 'MULTIFACTOR_FUSION_FREEZE_MANIFEST.json')
    if freeze['test_read_at_freeze_time']:
        raise SystemExit('freeze manifest says test was already read; refusing to re-run')
    packed, dyn = _method_packs('test', device=args.device)

    rows = []
    for pool in packed:
        for ds in DATASETS:
            for method, p in packed[pool][ds].items():
                rows.append({'dataset': ds, 'pool': pool, 'method': method,
                             **{k: p['metrics'][k] for k in METRIC_KEYS},
                             'n_users': len(p['ndcg20']),
                             'target_in_union': int(p['in_union'].sum())})
    pd.DataFrame(rows).to_csv(OUT / 'test' / 'results.csv', index=False)

    n = args.bootstrap
    SEED = 20260912
    boot_rows = []

    def _boot(a, b, name, ds, pool, ma, mb):
        r = paired_bootstrap(a, b, n=n, seed=SEED)
        boot_rows.append({'comparison': name, 'pool': pool, 'dataset': ds,
                          'method_a': ma, 'method_b': mb, 'metric': 'NDCG@20', **r})

    def add(name, pa, ma, pb, mb):
        for ds in DATASETS:
            _boot(packed[pa][ds][ma]['ndcg20'], packed[pb][ds][mb]['ndcg20'],
                  name, ds, f'{pa}_vs_{pb}', ma, mb)
        _boot(np.concatenate([packed[pa][ds][ma]['ndcg20'] for ds in DATASETS]),
              np.concatenate([packed[pb][ds][mb]['ndcg20'] for ds in DATASETS]),
              name, 'ALL_POOLED', f'{pa}_vs_{pb}', ma, mb)

    add('S4_Gate_vs_OldDynamicFusion', 'spectral4', 'Gate', 'spectral4', 'OldDynamicFusion')
    add('S4_LSE_vs_OldDynamicFusion', 'spectral4', 'LSE', 'spectral4', 'OldDynamicFusion')
    add('S4_Uniform_vs_OldDynamicFusion', 'spectral4', 'Uniform', 'spectral4',
        'OldDynamicFusion')
    add('S4_Gate_vs_S4_LSE', 'spectral4', 'Gate', 'spectral4', 'LSE')
    add('M5_Gate_vs_S4_Gate', 'multi5', 'Gate', 'spectral4', 'Gate')
    add('M5_LSE_vs_S4_LSE', 'multi5', 'LSE', 'spectral4', 'LSE')
    add('M5_Uniform_vs_S4_Uniform', 'multi5', 'Uniform', 'spectral4', 'Uniform')
    add('M5_Gate_vs_M5_LSE', 'multi5', 'Gate', 'multi5', 'LSE')
    add('M5_Gate_vs_OldDynamicFusion', 'multi5', 'Gate', 'multi5', 'OldDynamicFusion')
    add('M5_LSE_vs_OldDynamicFusion', 'multi5', 'LSE', 'multi5', 'OldDynamicFusion')


    pd.DataFrame(boot_rows).to_csv(OUT / 'test' / 'bootstrap.csv', index=False)
    write_json(OUT / 'test' / 'dynamic_fusion_test_replay.json',
               {p: {ds: {k: v for k, v in dyn[p][ds].items() if k != 'weights'}
                    for ds in dyn[p]} for p in dyn})
    print(f'wrote {OUT / "test" / "results.csv"} and bootstrap.csv')
    return None


# ---------------------------------------------------------------------------
# stage L+M — coverage and unique-factorization analysis
# ---------------------------------------------------------------------------
def stage_coverage(args):
    """Sections 51-53: candidate coverage, RQ-VAE added value, unique-F groups."""
    import pandas as pd
    from mf_eval import Cell
    from mf_core import rank_and_metrics
    from mf_eval import paired_bootstrap
    val = read_json(OUT / 'validation' / 'validation_baseline_selection.json')
    scalers = load_scalers()
    cov_rows, rq_rows, ov_rows = [], [], []

    for split in ('valid', 'test'):
        for ds in DATASETS:
            cells = {}
            for pool in ('spectral4', 'multi5'):
                cells[pool] = Cell(ds, pool, split, scalers,
                                   val['pools'][pool][ds]['lambda_selected'],
                                   device=args.device)
            c4, c5 = cells['spectral4'], cells['multi5']
            # --- 51. union recall, all three candidate universes ----------------
            for pool, c in (('spectral4', c4), ('multi5', c5)):
                cov_rows.append({'row': 'union', 'dataset': ds, 'split': split,
                                 'pool': pool, 'n_users': len(c.target),
                                 'n_target_in_union': int((c.tpos >= 0).sum()),
                                 'union_recall': float((c.tpos >= 0).mean())})
                for j, f in enumerate(c.f_list):
                    hit = 0
                    for u in range(len(c.target)):
                        s, e = c.starts[u], c.ends[u]
                        w = np.where(c.items[s:e] == c.target[u])[0]
                        if len(w) and c.present[s + int(w[0]), j]:
                            hit += 1
                    cov_rows.append({'row': 'single_factorization', 'dataset': ds,
                                     'split': split, 'pool': pool, 'factorisation': f,
                                     'n_users': len(c.target), 'n_target_in_union': hit,
                                     'union_recall': hit / len(c.target)})
            # --- DynamicFusion's own union, for the fairness comparison --------
            from mf_dynamic_fusion import dynamic_arrays
            da = dynamic_arrays(ds, split)
            tgt_raw = da['target']                       # 0-based rows
            dfu = da['items']
            hit = 0
            for u in range(len(tgt_raw)):
                if tgt_raw[u] >= 0 and (dfu[u] == tgt_raw[u]).any():
                    hit += 1
            cov_rows.append({'row': 'dynamicfusion_union', 'dataset': ds, 'split': split,
                             'pool': 'dynamicfusion_s4_union', 'n_users': len(tgt_raw),
                             'n_target_in_union': hit,
                             'union_recall': hit / len(tgt_raw)})
            # pairwise overlap between the two universes on the same users
            a = set()
            for u in range(min(len(c4.users), len(dfu))):
                s, e = c4.starts[u], c4.ends[u]
                for it in c4.items[s:e]:
                    a.add((u, int(it)))
            b = set()
            for u in range(len(dfu)):
                for it in dfu[u]:
                    if it >= 0:
                        b.add((u, int(it)))
            ov_rows.append({'dataset': ds, 'split': split,
                            's4_union_pairs': len(a), 'dynamic_union_pairs': len(b),
                            'intersection': len(a & b),
                            'jaccard': len(a & b) / max(len(a | b), 1)})
            # --- 52. RQ-only targets -------------------------------------------
            if split == 'test':
                idx = {f: j for j, f in enumerate(c5.f_list)}
                packs, _ = c5.packs()
                rq_only_mask = np.zeros(len(c5.target), dtype=bool)
                for u in range(len(c5.target)):
                    s, e = c5.starts[u], c5.ends[u]
                    w = np.where(c5.items[s:e] == c5.target[u])[0]
                    if not len(w):
                        continue
                    row = c5.present[s + int(w[0])]
                    rq = row[idx['rqvae3']]
                    sp = any(row[idx[f]] for f in S4)
                    rq_only_mask[u] = bool(rq and not sp)
                # NDCG contribution attributable to the fifth factorization: the
                # same method on M5 (which can see RQ's probability/NLL) versus S4
                # (which cannot), restricted to users whose target ONLY RQ retrieved.
                c4t = Cell(ds, 'spectral4', split, scalers,
                           val['pools']['spectral4'][ds]['lambda_selected'],
                           device=args.device)
                packs4, _ = c4t.packs()
                contrib = {}
                for name in ('Gate', 'LSE', 'Uniform'):
                    if name in packs and name in packs4 and rq_only_mask.sum() > 1:
                        contrib[name] = paired_bootstrap(
                            packs[name]['ndcg20'][rq_only_mask],
                            packs4[name]['ndcg20'][rq_only_mask])
                rq_rows.append({'dataset': ds, 'n_users': len(c5.target),
                                'rq_only': int(rq_only_mask.sum()),
                                'rq_only_fraction': float(rq_only_mask.mean()),
                                'm5_gate_ndcg20_on_rq_only':
                                    float(packs['Gate']['ndcg20'][rq_only_mask].mean())
                                    if rq_only_mask.any() else None,
                                'm5_lse_ndcg20_on_rq_only':
                                    float(packs['LSE']['ndcg20'][rq_only_mask].mean())
                                    if rq_only_mask.any() else None,
                                'm5_uniform_ndcg20_on_rq_only':
                                    float(packs['Uniform']['ndcg20'][rq_only_mask].mean())
                                    if rq_only_mask.any() else None,
                                's4_gate_ndcg20_on_rq_only':
                                    float(packs4['Gate']['ndcg20'][rq_only_mask].mean())
                                    if rq_only_mask.any() else None,
                                'm5_minus_s4_gate_on_rq_only':
                                    contrib.get('Gate', {}).get('delta'),
                                'm5_minus_s4_gate_ci':
                                    [contrib.get('Gate', {}).get('ci_low'),
                                     contrib.get('Gate', {}).get('ci_high')],
                                'm5_minus_s4_gate_p': contrib.get('Gate', {}).get('p')})
                del c4t, packs4
            del cells, c4, c5
    pd.DataFrame(cov_rows).to_csv(OUT / 'test' / 'coverage.csv', index=False)
    pd.DataFrame(ov_rows).to_csv(OUT / 'test' / 'union_overlap.csv', index=False)
    pd.DataFrame(rq_rows).to_csv(OUT / 'test' / 'rqvae_added_value.csv', index=False)

    # --- 53. unique-factorization target groups --------------------------------
    groups_rows = []
    for pool in ('spectral4', 'multi5'):
        for split in ('valid', 'test'):
            for ds in DATASETS:
                cell = Cell(ds, pool, split, scalers,
                            val['pools'][pool][ds]['lambda_selected'],
                            device=args.device)
                packs, _ = cell.packs()
                labels = np.empty(len(cell.target), dtype=object)
                for u in range(len(cell.target)):
                    s, e = cell.starts[u], cell.ends[u]
                    w = np.where(cell.items[s:e] == cell.target[u])[0]
                    if not len(w):
                        labels[u] = 'none'
                        continue
                    row = cell.present[s + int(w[0])]
                    src = [FLABEL[f] for j, f in enumerate(cell.f_list) if row[j]]
                    labels[u] = (f'Only-{src[0]}' if len(src) == 1 else
                                 ('multi-hit' if src else 'none'))
                for name in ('Uniform', 'LSE', 'Hybrid', 'Gate', 'GateLikelihood',
                             'OldDynamicFusion', 'Oracle'):
                    if name not in packs:
                        continue
                    v = packs[name]['ndcg20']
                    for g in sorted(set(labels.tolist())):
                        m = labels == g
                        groups_rows.append({
                            'dataset': ds, 'pool': pool, 'split': split,
                            'method': name, 'group': g, 'n': int(m.sum()),
                            'ndcg20': float(v[m].mean()) if m.any() else None})
                del cell
    pd.DataFrame(groups_rows).to_csv(OUT / 'test' / 'unique_factorisation_groups.csv',
                                     index=False)
    print('wrote coverage.csv, union_overlap.csv, rqvae_added_value.csv, '
          'unique_factorisation_groups.csv')
    return None


# ---------------------------------------------------------------------------
# stage M — efficiency
# ---------------------------------------------------------------------------
def stage_efficiency(args):
    """Section 54-56: compute/VRAM accounting, from the artifacts themselves."""
    import pandas as pd
    from mf_eval import Cell
    val = read_json(OUT / 'validation' / 'validation_baseline_selection.json')
    scalers = load_scalers()
    gate_info = read_json(OUT / 'validation' / 'gate_training.json')
    rows = []
    for split in ('valid', 'test'):
        for pool in ('spectral4', 'multi5'):
            for ds in DATASETS:
                df, _ = load_split(OUT, ds, pool, split)
                n_users = int(df.user.nunique())
                F = len(pool_factorizations(pool))
                rows += [
                    {'dataset': ds, 'split': split, 'pool': pool,
                     'quantity': 'candidate_union_size_mean',
                     'value': round(len(df) / n_users, 4), 'unit': 'items/user'},
                    {'dataset': ds, 'split': split, 'pool': pool,
                     'quantity': 'candidate_factorisation_pairs',
                     'value': int(len(df)), 'unit': 'pairs'},
                    {'dataset': ds, 'split': split, 'pool': pool,
                     'quantity': 'nll_sequences_total',
                     'value': int(len(df) * F), 'unit': 'teacher-forced sequences'},
                    {'dataset': ds, 'split': split, 'pool': pool,
                     'quantity': 'nll_sequences_per_user',
                     'value': round(len(df) * F / n_users, 2), 'unit': 'sequences/user'},
                ]
    for p in sorted((OUT / 'validation').glob('_VALID_*_NLL_PROVENANCE.json')) + \
             sorted((OUT / 'test').glob('_TEST_*_NLL_PROVENANCE.json')):
        j = read_json(p)
        try:
            split = j['split']
        except Exception:                                    # pragma: no cover
            split = 'unknown'
        for ds, cell in j['cells'].items():
            for f, v in cell.get('computed', {}).items():
                rows.append({'dataset': ds, 'split': split, 'pool': j['pool'],
                             'quantity': f'nll_scoring_seconds_{f}',
                             'value': v['elapsed_s'], 'unit': 's per dataset'}
                            )
            for f, v in cell.get('reused', {}).items():
                rows.append({'dataset': ds, 'split': split, 'pool': j['pool'],
                             'quantity': f'nll_reused_seconds_saved_{f}',
                             'value': float(v.get('n_reused', 0)),
                             'unit': 'values reused (not recomputed)'})
    for pool, g in gate_info.items():
        rows += [
            {'dataset': 'ALL', 'split': 'valid', 'pool': pool,
             'quantity': 'gate_parameters', 'value': g['n_params'], 'unit': 'params'},
            {'dataset': 'ALL', 'split': 'valid', 'pool': pool,
             'quantity': 'gate_best_epoch', 'value': g['best_epoch'], 'unit': 'epoch'},
            {'dataset': 'ALL', 'split': 'valid', 'pool': pool,
             'quantity': 'gate_d_in', 'value': g['d_in'], 'unit': 'features'},
        ]
    pf = read_json(OUT / 'OOM_PREFLIGHT.json')
    rows += [
        {'dataset': 'Beauty', 'split': 'valid', 'pool': 'multi5',
         'quantity': 'preflight_peak_vram_reserved', 'value': pf['peak_reserved_mib'],
         'unit': 'MiB'},
        {'dataset': 'Beauty', 'split': 'valid', 'pool': 'multi5',
         'quantity': 'preflight_peak_vram_fraction', 'value':
             round(pf['vram_peak_fraction_of_total'], 4), 'unit': 'of total VRAM'},
        {'dataset': 'Beauty', 'split': 'valid', 'pool': 'multi5',
         'quantity': 'predeclared_candidate_chunk', 'value': pf['cand_chunk_predeclared'],
         'unit': 'sequences per forward pass'},
    ]
    pd.DataFrame(rows).to_csv(OUT / 'test' / 'efficiency.csv', index=False)
    print(f'wrote {OUT / "test" / "efficiency.csv"} ({len(rows)} rows)')
    return None


def gates_ok():
    p = OUT / 'validation' / 'gate_training.json'
    return list(read_json(p).keys()) if p.exists() else []


if __name__ == '__main__':
    sys.exit(main())
