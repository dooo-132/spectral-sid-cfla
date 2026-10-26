#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP D/E/F/G — apply the FROZEN development gate to Sports, zero-shot.

Nothing is trained here.  The gate is loaded from
`outputs/disagreement_gate/runs/mlp_lambda0_2025_370bc979/gate.pt` with its weights
untouched, the 46-feature schema is the frozen one, the scoring formulas are the frozen
ones, and the only Sports-specific decisions are the two the harness pre-registered:

  * unseen-domain one-hot   : dataset one-hot = [0,0,0,0]  (section 22)
  * domain-transfer scaling : the global frozen scaler pooled from the four development
                              datasets' RouterTrain rows  (section 23)

Order of operations: `freeze` writes SPORTS_CONFIRMATION_FREEZE_MANIFEST.json, and only
`test` (a separate subcommand) reads the Sports test split.

Subcommands
  freeze      build VALID features, evaluate the frozen method on VALID, write the manifest
  test        (after the freeze) build TEST features and evaluate every method

Usage
  python3 sc_gate_eval.py freeze
  python3 sc_gate_eval.py test
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
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO),
           str(REPO / 'analysis' / 'mechanism' / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET = 'Sports_and_Outdoors'
import os
OUT = Path(os.environ.get('SC_OUT', REPO / 'outputs' / 'sports_confirmation'))
GATE_DIR = REPO / 'outputs' / 'disagreement_gate'
TRAIN_RUN = OUT / 'runs'
BOOT_N = 5000
BOOT_SEED = 20260912
KS = (5, 10, 20)
RUNGS = ('Only_F1', 'Only_F2', 'Both', 'Neither')


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load_frozen():
    """Gate checkpoint, feature schema and scalers, all read-only."""
    from dg_features import DATASET_ONEHOT, FEATURES
    from dg_train_gate import MLPGate, LinearGate
    man = json.loads((GATE_DIR / 'GATE_FREEZE_MANIFEST.json').read_text())
    tag = man['gate']['name']
    run = Path(man['gate']['run_directory'])
    ck = torch.load(run / 'gate.pt', map_location='cpu')
    model = MLPGate(ck['d_in']) if ck['kind'] == 'mlp' else LinearGate(ck['d_in'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    gsc = json.loads((OUT / 'valid' / 'GLOBAL_FEATURE_SCALER.json').read_text())
    return {'model': model, 'ck': ck, 'manifest': man, 'tag': tag,
            'features': FEATURES, 'onehot': DATASET_ONEHOT, 'global_scaler': gsc}


def build_features(split, G):
    """46 frozen features for Sports, scaled with the global frozen scaler."""
    from dg_features import (DATASET_ONEHOT, FEATURES, USER_LEVEL, per_user_features)
    u = pd.read_parquet(OUT / split / 'candidate_union.parquet')
    n = pd.read_parquet(OUT / split / 'candidate_nll.parquet',
                        columns=['TotalNLL_F1', 'TotalNLL_F2'])
    sc = json.loads((OUT / 'valid' / 'candidate_nll_scaler.json').read_text())
    z1_all = (n.TotalNLL_F1.values - sc['mu_F1']) / sc['sigma_F1']
    z2_all = (n.TotalNLL_F2.values - sc['mu_F2']) / sc['sigma_F2']

    users = np.unique(u.user.values)
    starts = np.searchsorted(u.user.values, users)
    ends = np.append(starts[1:], len(u))
    parts = {k: [] for k in FEATURES}
    U_all, L_all = [], []
    P1_all, P2_all = [], []
    for s, e in zip(starts, ends):
        items = u.candidate_item.values[s:e].astype(np.int64)
        p1 = u.retrieval_score_F1.values[s:e].astype(np.float64)
        p2 = u.retrieval_score_F2.values[s:e].astype(np.float64)
        r1 = u.rank_F1_if_present.values[s:e].astype(np.int64)
        r2 = u.rank_F2_if_present.values[s:e].astype(np.int64)
        c, usc, lse = per_user_features(items, p1, p2, r1, r2, z1_all[s:e], z2_all[s:e])
        for k in FEATURES:
            # `per_user_features` emits the retrieval / likelihood / disagreement /
            # user-level blocks; the dataset one-hot is materialised here and is all
            # zeros for Sports (unseen domain, harness section 22)
            parts[k].append(c[k] if k in c else np.zeros(e - s))
        U_all.append(usc); L_all.append(lse)
        P1_all.append(p1); P2_all.append(p2)
    X = np.column_stack([np.concatenate(parts[k]) for k in FEATURES]).astype(np.float64)
    # unseen domain: every one-hot is zero (section 22)
    for k in DATASET_ONEHOT:
        X[:, FEATURES.index(k)] = 0.0
    # global frozen scaler on the continuous columns
    cont = G['global_scaler']['continuous_idx']
    X[:, cont] = (X[:, cont] - np.asarray(G['global_scaler']['global_mean'])) / \
                 np.asarray(G['global_scaler']['global_std'])
    if not np.isfinite(X).all():
        raise AssertionError('non-finite feature values for Sports')
    # target as a 0-based SID row, from the canonical parquet target (item -> row-1)
    tgt_item = pd.read_parquet(_ART_ROOT + '/data' / Path(DATASET) /
                               f'{split}.parquet')['target'].to_numpy()
    tgt_row = tgt_item.astype(np.int64) - 1
    return {'X': X.astype(np.float32), 'users': users, 'starts': starts, 'ends': ends,
            'U': np.concatenate(U_all), 'L': np.concatenate(L_all),
            # per-factorization retrieval probabilities, needed for the BestF1/BestF2
            # baselines.  `U` is the Uniform score and `L` the LSE score, so ranking
            # BestF1 by `U` would silently make it identical to Uniform.
            'P1': np.concatenate(P1_all), 'P2': np.concatenate(P2_all),
            'cand_item': u.candidate_item.values.astype(np.int64),
            'z_target': tgt_row, 'features': FEATURES, 'n_users': len(users)}


def z_by_user(x, starts, ends):
    out = np.empty_like(x)
    for s, e in zip(starts, ends):
        seg = x[s:e]; mu = seg.mean(); sd = seg.std(ddof=0)
        out[s:e] = (seg - mu) / sd if sd > 1e-8 else 0.0
    return out


def gate_values(G, F):
    """Frozen gate applied to Sports candidates; returned flat."""
    W = int((F['ends'] - F['starts']).max())
    n = F['n_users']
    Pad = np.zeros((n, W, F['X'].shape[1]), dtype=np.float32)
    for i in range(n):
        s, e = int(F['starts'][i]), int(F['ends'][i])
        Pad[i, :e - s] = F['X'][s:e]
    with torch.no_grad():
        g = G['model'](torch.tensor(Pad)).numpy()
    out = np.empty(len(F['X']), dtype=np.float64)
    for i in range(n):
        s, e = int(F['starts'][i]), int(F['ends'][i])
        out[s:e] = g[i, :e - s]
    return out


def ranks_and_scores(F, gate):
    ZU = z_by_user(F['U'], F['starts'], F['ends'])
    ZL = z_by_user(F['L'], F['starts'], F['ends'])
    S = (1 - gate) * ZU + gate * ZL
    n = F['n_users']
    tgt_rel = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        # z_target is indexed by USER POSITION (the parquet has one row per user and the
        # candidate union preserves that order), not by user id
        t = int(F['z_target'][i]) if i < len(F['z_target']) else -1
        s, e = int(F['starts'][i]), int(F['ends'][i])
        hit = np.where(F['cand_item'][s:e] == t)[0]
        if len(hit):
            tgt_rel[i] = int(hit[0])

    def rank_under(score):
        rk = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            r = tgt_rel[i]
            if r < 0:
                continue
            s, e = int(F['starts'][i]), int(F['ends'][i])
            seg = score[s:e]
            rk[i] = 1 + int((seg > seg[r]).sum())
        return rk

    return {'ZU': ZU, 'ZL': ZL, 'S': S, 'tgt_rel': tgt_rel,
            'r_uniform': rank_under(ZU), 'r_lse': rank_under(ZL),
            'r_gate': rank_under(S),
            'r_f1': rank_under(F['P1']), 'r_f2': rank_under(F['P2']),
            'r_oracle': np.where(tgt_rel >= 0, 1, -1)}


def summarize(rank, ks=KS):
    out = {}
    for k in ks:
        hit = (rank >= 1) & (rank <= k)
        out[f'Recall@{k}'] = float(hit.mean())
        v = np.zeros(len(rank)); v[hit] = 1.0 / np.log2(rank[hit] + 1)
        out[f'NDCG@{k}'] = float(v.mean())
    return out


def ndcg_vec(rank, k=20):
    r = np.asarray(rank); hit = (r >= 1) & (r <= k)
    v = np.zeros(len(r)); v[hit] = 1.0 / np.log2(r[hit] + 1)
    return v


def paired_bootstrap(a, b, n=BOOT_N, seed=BOOT_SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = len(d); delta = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, m, size=(n, m))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * float((boots <= 0).mean()) if delta >= 0 else 2 * float((boots >= 0).mean())
    return {'delta': delta, 'ci_low': float(lo), 'ci_high': float(hi),
            'p_value': min(1.0, max(p, 1.0 / n)), 'n_users': m}


def cmd_freeze(a):
    G = load_frozen()
    F = build_features('valid', G)
    gate = gate_values(G, F)
    R = ranks_and_scores(F, gate)
    rows = [{'dataset': DATASET, 'method': m, 'n': F['n_users'],
             **summarize(r)}
            for m, r in (('BestF1', R['r_f1']), ('BestF2', R['r_f2']),
                         ('Uniform', R['r_uniform']), ('LSE', R['r_lse']),
                         ('Gate', R['r_gate']), ('Oracle', R['r_oracle']))]
    pd.DataFrame(rows).to_csv(OUT / 'valid' / 'validation_metrics.csv', index=False)
    print('[valid] frozen method on Sports VALID:')
    for r in rows:
        print(f"  {r['method']:8s} NDCG@20={r['NDCG@20']:.6f}  Recall@20={r['Recall@20']:.6f}")

    man = {
        'study': 'pristine Sports confirmation of the frozen candidate-conditioned method',
        'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                  text=True, cwd=str(REPO)).stdout.strip(),
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'sports_spectral_k3_sid_sha256': sha256_file(OUT / 'sid' / 'spectral_k3' / 'sid.npy'),
        'sports_rqvae_sid_sha256': sha256_file(
            Path(_ART_ROOT + '/data') / DATASET / f'{DATASET}_t5_rqvae.npy'),
        'shared_checkpoint_sha256': sha256_file(a.ckpt),
        'training_config_sha256': (sha256_file(TRAIN_RUN / a.run_id / 'config.json')
                                   if (TRAIN_RUN / a.run_id / 'config.json').exists()
                                   else None),
        'training_run_id': a.run_id,
        'valid_nll_scaler': json.loads((OUT / 'valid' / 'candidate_nll_scaler.json').read_text()),
        'frozen_uniform_formula': G['manifest']['frozen_score_definitions']['Uniform'],
        'frozen_lse_formula': G['manifest']['frozen_score_definitions']['LSE'],
        'frozen_mix_formula': G['manifest']['frozen_score_definitions']['mix'],
        'frozen_z_norm': G['manifest']['frozen_score_definitions']['z_norm'],
        'frozen_gate_checkpoint_sha256': G['manifest']['gate']['checkpoint_sha256'],
        'frozen_gate_name': G['tag'],
        'frozen_gate_run_directory': G['manifest']['gate']['run_directory'],
        'feature_schema_sha256': hashlib.sha256(
            json.dumps(G['features']).encode()).hexdigest(),
        'feature_count': len(G['features']),
        'sports_unseen_domain_onehot_rule': '[0,0,0,0] — no fifth dimension added',
        'global_feature_scaler_sha256': sha256_file(
            OUT / 'valid' / 'GLOBAL_FEATURE_SCALER.json'),
        'global_feature_scaler_built_from': G['global_scaler']['fitted_on'],
        'beam_size': 30,
        'candidate_union_rule': 'C_F1(u) ∪ C_F2(u) from generation output only; target never added',
        'valid_metrics': rows,
        'sports_test_read': False,
    }
    (OUT / 'SPORTS_CONFIRMATION_FREEZE_MANIFEST.json').write_text(
        json.dumps(man, indent=2, default=str))
    print(f'[freeze] wrote SPORTS_CONFIRMATION_FREEZE_MANIFEST.json')
    return 0


def cmd_test(a):
    fp = OUT / 'SPORTS_CONFIRMATION_FREEZE_MANIFEST.json'
    if not fp.exists():
        raise SystemExit('refusing to read test before the freeze manifest exists')
    man = json.loads(fp.read_text())
    if man.get('sports_test_read') is not False:
        raise SystemExit('freeze manifest does not record sports_test_read = false')
    G = load_frozen()
    if sha256_file(Path(G['manifest']['gate']['run_directory']) / 'gate.pt') != \
            man['frozen_gate_checkpoint_sha256']:
        raise SystemExit('frozen gate checkpoint hash changed since the freeze')

    F = build_features('test', G)
    gate = gate_values(G, F)
    R = ranks_and_scores(F, gate)
    methods = {'BestF1': R['r_f1'], 'BestF2': R['r_f2'], 'Uniform': R['r_uniform'],
               'LSE': R['r_lse'], 'Gate': R['r_gate'], 'Oracle': R['r_oracle']}
    rows = [{'dataset': DATASET, 'method': m, 'n': F['n_users'], **summarize(r)}
            for m, r in methods.items()]
    pd.DataFrame(rows).to_csv(OUT / 'test' / 'test_results.csv', index=False)

    boot = []
    for k in KS:
        for comp, other in (('Gate_vs_Uniform', 'Uniform'), ('Gate_vs_LSE', 'LSE'),
                            ('LSE_vs_Uniform', 'Uniform')):
            st = paired_bootstrap(ndcg_vec(methods['Gate' if comp.startswith('Gate') else 'LSE'], k),
                                  ndcg_vec(methods[other], k))
            boot.append({'comparison': comp, 'dataset': DATASET, 'metric': f'NDCG@{k}', **st})
    pd.DataFrame(boot).to_csv(OUT / 'test' / 'bootstrap.csv', index=False)

    # subgroups (evaluation only)
    u = pd.read_parquet(OUT / 'test' / 'candidate_union.parquet')
    grp = np.full(F['n_users'], 'Neither', dtype=object)
    for i in range(F['n_users']):
        r = int(R['tgt_rel'][i])
        if r < 0:
            continue
        s = int(F['starts'][i])
        h1 = 1 <= int(u.rank_F1_if_present.values[s + r]) <= 20
        h2 = 1 <= int(u.rank_F2_if_present.values[s + r]) <= 20
        grp[i] = 'Both' if (h1 and h2) else ('Only_F1' if h1 else
                                             ('Only_F2' if h2 else 'Neither'))
    srows = []
    for g in RUNGS:
        m = grp == g
        if m.sum() == 0:
            continue
        for name, r in (('Uniform', R['r_uniform']), ('LSE', R['r_lse']),
                        ('Gate', R['r_gate']), ('Oracle', R['r_oracle'])):
            ss = summarize(r[m], (20,))
            srows.append({'dataset': DATASET, 'group': g, 'n': int(m.sum()),
                          'method': name, 'NDCG@20': ss['NDCG@20'],
                          'Recall@20': ss['Recall@20']})
    pd.DataFrame(srows).to_csv(OUT / 'test' / 'subgroup_results.csv', index=False)

    # mechanism diagnostics: target vs non-target LSE AUC
    mrows = []
    num = den = 0
    for i in range(F['n_users']):
        r = int(R['tgt_rel'][i])
        if r < 0:
            continue
        s, e = int(F['starts'][i]), int(F['ends'][i])
        seg = R['ZL'][s:e]
        ts = seg[r]
        others = np.delete(seg, r)
        if len(others) == 0:
            continue
        num += float((ts > others).sum()); den += len(others)
    mrows.append({'dataset': DATASET, 'metric': 'target_vs_nontarget_LSE_AUC',
                  'value': (num / den) if den else float('nan'), 'n_pairs': den})
    for name, r in (('Uniform', R['r_uniform']), ('LSE', R['r_lse']), ('Gate', R['r_gate'])):
        mrows.append({'dataset': DATASET, 'metric': f'{name}_NDCG@20',
                      'value': summarize(r, (20,))['NDCG@20'], 'n_pairs': F['n_users']})
    for g in RUNGS:
        mrows.append({'dataset': DATASET, 'metric': f'group_fraction_{g}',
                      'value': float((grp == g).mean()), 'n_pairs': int((grp == g).sum())})
    mrows.append({'dataset': DATASET, 'metric': 'mean_gate',
                  'value': float(gate.mean()), 'n_pairs': len(gate)})
    # union coverage = fraction of users whose target the generator reached at all.
    # This is the reranking ceiling and must be measured, not inferred from subgroup
    # counts (a Neither user can still have the target present outside the top-20).
    in_union = R['tgt_rel'] >= 0
    mrows.append({'dataset': DATASET, 'metric': 'union_coverage',
                  'value': float(in_union.mean()), 'n_pairs': int(in_union.sum())})
    mrows.append({'dataset': DATASET, 'metric': 'n_users',
                  'value': float(F['n_users']), 'n_pairs': F['n_users']})
    pd.DataFrame(mrows).to_csv(OUT / 'test' / 'mechanism_diagnostics.csv', index=False)

    man['sports_test_read'] = True
    man['test_evaluated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fp.write_text(json.dumps(man, indent=2, default=str))

    print('[test] Sports results:')
    for r in rows:
        print(f"  {r['method']:8s} Recall@20={r['Recall@20']:.6f} NDCG@20={r['NDCG@20']:.6f}")
    print()
    for b in boot:
        if b['metric'] == 'NDCG@20':
            print(f"  {b['comparison']:18s} delta={b['delta']:+.6f} "
                  f"CI=[{b['ci_low']:+.6f},{b['ci_high']:+.6f}] p={b['p_value']:.4f}")
    print()
    for r in srows:
        print(f"  {r['group']:8s} n={r['n']:>6d} {r['method']:8s} NDCG@20={r['NDCG@20']:.6f}")
    np.savez_compressed(OUT / 'test' / 'per_user_ranks.npz',
                        r_uniform=R['r_uniform'], r_lse=R['r_lse'], r_gate=R['r_gate'],
                        r_f1=R['r_f1'], r_f2=R['r_f2'], r_oracle=R['r_oracle'],
                        group=grp, gate=gate, tgt_rel=R['tgt_rel'])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--ckpt', required=True)
    f.add_argument('--run_id', required=True)
    f.set_defaults(func=cmd_freeze)
    t = sub.add_parser('test')
    t.set_defaults(func=cmd_test)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main())
