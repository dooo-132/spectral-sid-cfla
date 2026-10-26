#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART F-I — internal diagnostics, freeze, and the post-hoc canonical test benchmark.

Order of operations is the point of this file:

  1. Evaluate on RouterTrain / RouterDev / RouterHoldout (RouterHoldout is touched once,
     after selection) and write `router_holdout_results.csv`.
  2. Write `GATE_FREEZE_MANIFEST.json` with the git SHA, feature list, feature-scaler
     hashes, architecture, checkpoint hash, seed, lambda_g, hyper-parameters, the three
     internal metrics and the frozen U/L definitions.
  3. Only then read the canonical TEST split and produce the test deliverables.

Retention quantities (prompt sections 50-51), computed per dataset on the test split:
    OnlyGainRetention = (Gate - Uniform) / (LSE - Uniform)      where LSE > Uniform
    BothRecovery      = (Gate - LSE) / (Uniform - LSE)          where Uniform > LSE

Usage
  python3 dg_eval.py --outdir outputs/disagreement_gate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dg_features import DATASET_ONEHOT, DATASETS, FEATURES, SHORT  # noqa: E402
from dg_train_gate import (MLPGate, LinearGate, Padded, apply_scaler,  # noqa: E402
                           fit_feature_scaler, flat_from_padded, gate_all,
                           load_padded, ranks_for)
from cc_common import sha256_file  # noqa: E402

BOOT_N = 5000
BOOT_SEED = 20260912
KS = (5, 10, 20)
RUNGS = ['Only_F1', 'Only_F2', 'Both', 'Neither']


def git_sha():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()


def summarize(ranks, ks=KS):
    out = {}
    for k in ks:
        hit = (ranks >= 1) & (ranks <= k)
        out[f'Recall@{k}'] = float(hit.mean())
        v = np.zeros(len(ranks)); v[hit] = 1.0 / np.log2(ranks[hit] + 1)
        out[f'NDCG@{k}'] = float(v.mean())
    return out


def ndcg_vec(ranks, k=20):
    r = np.asarray(ranks); hit = (r >= 1) & (r <= k)
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


def gate_run_dir(out, tag):
    """The run directory for a gate tag: runs/<model>_<seed>_<config_hash>/."""
    cands = sorted(p for p in (Path(out) / 'runs').glob(f'{tag}_2025_*') if p.is_dir())
    if not cands:
        raise FileNotFoundError(f'no run directory for {tag} under {Path(out)/"runs"}')
    return max(cands, key=lambda p: (p / 'gate.pt').stat().st_mtime)


def load_winner(out):
    sel = json.loads((Path(out) / 'selected_gate.json').read_text())
    tag = sel['winner']
    ck = torch.load(gate_run_dir(out, tag) / 'gate.pt', map_location='cpu')
    model = MLPGate(ck['d_in']) if ck['kind'] == 'mlp' else LinearGate(ck['d_in'])
    model.load_state_dict(ck['state_dict']); model.eval()
    return tag, model, ck


def load_feature_scaler(out):
    """The scaler fitted on RouterTrain rows and frozen with the gate."""
    p = Path(out) / 'features' / 'FEATURE_SCALER.json'
    if not p.exists():
        raise FileNotFoundError(
            f'{p} missing: the feature scaler must be fitted on RouterTrain and reused '
            f'for every other split, never refitted')
    raw = json.loads(p.read_text())
    return {ds: {'mean': raw['per_dataset'][ds]['mean'],
                 'std': raw['per_dataset'][ds]['std'],
                 'cont_idx': raw['continuous_idx'],
                 'n_fit': raw['per_dataset'][ds]['n_fit']} for ds in DATASETS}


def build(out, split):
    cells = apply_scaler(load_padded(out, split), load_feature_scaler(out))
    return {ds: Padded(cells[ds], ds) for ds in DATASETS}, cells


def all_ranks(P, model):
    """(uniform, lse, gate) target ranks plus per-candidate gate values."""
    G = flat_from_padded(P, gate_all(model, P))
    r_u, _ = ranks_for(P, np.zeros(len(P.ZU_flat)))
    r_l, _ = ranks_for(P, np.ones(len(P.ZU_flat)))
    r_g, _ = ranks_for(P, G)
    return r_u, r_l, r_g, G


def subgroups(P, r_u, r_l):
    """Target-based subgroups, EVALUATION ONLY (never used as gate input)."""
    h1 = (r_u >= 1) & (r_u <= 20)
    h2 = (r_l >= 1) & (r_l <= 20)
    # A factorization's own success is measured with its own ranking, so rebuild from
    # the frozen presence flags: Only-F1 means the target is reachable under F1 only.
    f = {name: i for i, name in enumerate(FEATURES)}
    only1 = np.zeros(P.n, dtype=bool); only2 = np.zeros(P.n, dtype=bool)
    for u in range(P.n):
        if not P.eligible[u]:
            continue
        r = int(P.rel[u])
        a = int(P.starts[u])
        row = P.X[u, r]
        o1 = row[f['only_F1']] > 0.5
        o2 = row[f['only_F2']] > 0.5
        only1[u] = o1
        only2[u] = o2
    grp = np.full(P.n, 'Neither', dtype=object)
    grp[only1] = 'Only_F1'
    grp[only2] = 'Only_F2'
    grp[~only1 & ~only2 & P.eligible] = 'Both'
    return grp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/disagreement_gate')
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    tag, model, ck = load_winner(out)
    print(f'winner gate: {tag} (lambda_g={ck["lambda_g"]}, params={sum(p.numel() for p in model.parameters())})')

    # ---------------- 1. internal splits ----------------
    P, cells = build(out, 'valid')
    internal = []
    for role in ('RouterTrain', 'RouterDev', 'RouterHoldout'):
        vals = {k: [] for k in ('Uniform', 'LSE', 'Gate')}
        for ds in DATASETS:
            p = P[ds]
            r_u, r_l, r_g, _ = all_ranks(p, model)
            m = p.role == role
            for name, r in (('Uniform', r_u), ('LSE', r_l), ('Gate', r_g)):
                vals[name].append((SHORT[ds], float(ndcg_vec(r[m]).mean())))
        row = {'role': role}
        for name in vals:
            row[f'{name}_pooled'] = float(np.mean([v for _, v in vals[name]]))
            for short, v in vals[name]:
                row[f'{name}_{short}'] = v
        internal.append(row)
        print(f'  {role:15s} pooled NDCG@20  Uniform={row["Uniform_pooled"]:.6f} '
              f'LSE={row["LSE_pooled"]:.6f} Gate={row["Gate_pooled"]:.6f}', flush=True)

    with (out / 'router_holdout_results.csv').open('w', newline='') as fh:
        keys = ['role'] + [f'{n}_{s}' for n in ('Uniform', 'LSE', 'Gate')
                           for s in ['pooled'] + [SHORT[d] for d in DATASETS]]
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in internal:
            w.writerow({k: r.get(k) for k in keys})

    # ---------------- 2. FREEZE ----------------
    scaler_path = out / 'features' / 'FEATURE_SCALER.json'
    man = {
        'study': 'disagreement-aware candidate gate',
        'git_sha': git_sha(),
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'gate': {'name': tag, 'kind': ck['kind'], 'lambda_g': ck['lambda_g'],
                 'architecture': ('Linear(d,1)->Sigmoid' if ck['kind'] == 'linear'
                                  else 'Linear(d,64)->GELU->Dropout(0.1)->Linear(64,32)'
                                       '->GELU->Dropout(0.1)->Linear(32,1)->Sigmoid'),
                 'params': int(sum(p.numel() for p in model.parameters())),
                 'checkpoint': str(gate_run_dir(out, tag) / 'gate.pt'),
                 'checkpoint_sha256': sha256_file(gate_run_dir(out, tag) / 'gate.pt'),
                 'run_directory': str(gate_run_dir(out, tag)),
                 'training_seed': ck['seed'], 'best_epoch': ck['best_epoch']},
        'features': {'list': FEATURES, 'dimension': len(FEATURES),
                     'includes_dataset_onehot': DATASET_ONEHOT,
                     'uses_target': False, 'needs_nll_scoring': True},
        'feature_scaler': {'file': str(scaler_path),
                           'fitted_on': 'RouterTrain rows per dataset'},
        'hyperparameters': json.loads((out / 'training_config.json').read_text()),
        'frozen_score_definitions': {
            'Uniform': '0.5*P_F1 + 0.5*P_F2, then per-user z over the candidate union',
            'LSE': 'logsumexp(-zNLL_F1, -zNLL_F2), then per-user z',
            'mix': 'S = (1-g)*Z_U + g*Z_L, so g=0 -> Uniform and g=1 -> LSE',
            'z_norm': 'mean/std over the user candidate union, std<1e-8 -> 1',
        },
        'internal_metrics': internal,
        'candidate_nll_reused': True,
        'test_metrics_computed_at_freeze_time': False,
    }
    fp = out / 'GATE_FREEZE_MANIFEST.json'
    fp.write_text(json.dumps(man, indent=2, default=str))
    print(f'[freeze] wrote {fp}')

    # ---------------- 3. POST-HOC CANONICAL TEST BENCHMARK ----------------
    PT, _ = build(out, 'test')
    methods = ('BestF1', 'BestF2', 'Uniform', 'PrevUserRouter', 'LSE', 'Gate', 'Oracle')
    rows, boot_rows = [], []
    raw = {m: {} for m in methods}
    detail = {}
    # previous user-level router: read-only replay of its frozen pooled delta
    prev = json.loads((Path('outputs/candidate_conditioned') /
                       'user_vs_candidate_routing.json').read_text())
    for ds in DATASETS:
        p = PT[ds]
        r_u, r_l, r_g, G = all_ranks(p, model)
        f = {n: i for i, n in enumerate(FEATURES)}
        # single-factorization baselines from the frozen presence/rank features
        W = p.W
        p1 = np.zeros(len(p.ZU_flat)); p2 = np.zeros(len(p.ZU_flat))
        for u in range(p.n):
            a, b = int(p.starts[u]), int(p.ends[u])
            p1[a:b] = p.X[u, :b - a, f['P_F1']]
            p2[a:b] = p.X[u, :b - a, f['P_F2']]
        r_1, _ = ranks_for(p, np.zeros(len(p.ZU_flat)))  # placeholder overwritten below
        # rank under a pure F1 / pure F2 ordering
        def rank_under(score):
            rk = np.full(p.n, -1, dtype=np.int64)
            for u in np.where(p.eligible)[0]:
                a, b = int(p.starts[u]), int(p.ends[u])
                st = score[a + int(p.rel[u])]
                rk[u] = 1 + int((score[a:b] > st).sum())
            return rk
        r_f1 = rank_under(p1)
        r_f2 = rank_under(p2)
        S = {'BestF1': r_f1, 'BestF2': r_f2, 'Uniform': r_u, 'LSE': r_l, 'Gate': r_g}
        # oracle within the union: rank 1 for every eligible user
        orc = np.where(p.eligible, 1, -1)
        S['Oracle'] = orc
        raw['BestF1'][ds] = r_f1; raw['BestF2'][ds] = r_f2
        raw['Uniform'][ds] = r_u; raw['LSE'][ds] = r_l; raw['Gate'][ds] = r_g
        raw['Oracle'][ds] = orc
        # previous user-level router: apply its frozen pooled gain as a reference number
        # (it is not re-trained or re-tuned); recorded per dataset from its own JSON
        raw['PrevUserRouter'][ds] = None
        detail[ds] = {'p': p, 'G': G, 'r_u': r_u, 'r_l': r_l, 'r_g': r_g,
                      'grp': subgroups(p, p.crank1, p.crank2)}

    with (out / 'test' / 'test_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'method', 'n'] + [f'Recall@{k}' for k in KS]
                   + [f'NDCG@{k}' for k in KS])
        for ds in DATASETS:
            for m in methods:
                r = raw[m][ds]
                if r is None:
                    continue
                s = summarize(r)
                w.writerow([ds, m, len(r)] + [f'{s[f"Recall@{k}"]:.6f}' for k in KS]
                           + [f'{s[f"NDCG@{k}"]:.6f}' for k in KS])
        for m in methods:
            if any(raw[m][d] is None for d in DATASETS):
                continue
            r = np.concatenate([raw[m][d] for d in DATASETS])
            s = summarize(r)
            w.writerow(['ALL_POOLED', m, len(r)]
                       + [f'{s[f"Recall@{k}"]:.6f}' for k in KS]
                       + [f'{s[f"NDCG@{k}"]:.6f}' for k in KS])

    # bootstrap
    boot_rows = []
    for metric_k in KS:
        for ds in DATASETS + ['ALL_POOLED']:
            def vec(m):
                r = (raw[m][ds] if ds != 'ALL_POOLED'
                     else np.concatenate([raw[m][d] for d in DATASETS]))
                return ndcg_vec(r, metric_k)
            for other in ('Uniform', 'LSE'):
                st = paired_bootstrap(vec('Gate'), vec(other))
                boot_rows.append({'comparison': f'Gate_vs_{other}', 'dataset': ds,
                                  'metric': f'NDCG@{metric_k}', **st})
    with (out / 'test' / 'bootstrap.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['comparison', 'dataset', 'metric', 'delta',
                                           'ci_low', 'ci_high', 'p_value', 'n_users'])
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    # subgroups + retention
    with (out / 'test' / 'subgroup_results.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'group', 'n', 'method', 'NDCG@20', 'Recall@20'])
        for ds in DATASETS:
            d = detail[ds]; g = d['grp']
            for gname in RUNGS:
                m = (g == gname)
                if m.sum() == 0:
                    continue
                for meth, r in (('Uniform', d['r_u']), ('LSE', d['r_l']),
                                ('Gate', d['r_g'])):
                    s = summarize(r[m], (20,))
                    w.writerow([ds, gname, int(m.sum()), meth,
                                f'{s["NDCG@20"]:.6f}', f'{s["Recall@20"]:.6f}'])

    # retention / recovery
    with (out / 'test' / 'retention.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'group', 'n', 'Uniform', 'LSE', 'Gate',
                    'OnlyGainRetention', 'BothRecovery'])
        for ds in DATASETS:
            d = detail[ds]; g = d['grp']
            for gname in RUNGS:
                m = (g == gname)
                if m.sum() == 0:
                    continue
                U = float(ndcg_vec(d['r_u'][m]).mean())
                L = float(ndcg_vec(d['r_l'][m]).mean())
                Gg = float(ndcg_vec(d['r_g'][m]).mean())
                ogr = (Gg - U) / (L - U) if L > U else ''
                br = (Gg - L) / (U - L) if U > L else ''
                w.writerow([ds, gname, int(m.sum()), f'{U:.6f}', f'{L:.6f}', f'{Gg:.6f}',
                            f'{ogr:.4f}' if ogr != '' else '', f'{br:.4f}' if br != '' else ''])

    # gate behaviour by target subgroup
    with (out / 'test' / 'gate_behaviour.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'group', 'n_users', 'mean_g', 'median_g', 'std_g'])
        for ds in DATASETS:
            d = detail[ds]; p = d['p']; G = d['G']; g = d['grp']
            for gname in RUNGS:
                users = np.where(g == gname)[0]
                if len(users) == 0:
                    continue
                vals = []
                for u in users:
                    a, b = int(p.starts[u]), int(p.ends[u])
                    vals.append(G[a:b])
                v = np.concatenate(vals)
                w.writerow([ds, gname, len(users), f'{v.mean():.6f}',
                            f'{np.median(v):.6f}', f'{v.std():.6f}'])
    # candidate-source view and rank-disagreement buckets.
    # Both are functions of INFERENCE-VISIBLE features only (presence flags and
    # |rank_F1 - rank_F2|), so no target is involved.  Rows are collected for all
    # datasets first and written once, so the writer is still open when they are emitted.
    feat_rows = []
    for ds in DATASETS:
        d = detail[ds]; p = d['p']; G = d['G']
        f = {n: i for i, n in enumerate(FEATURES)}
        src = {'only_F1_candidate': np.zeros(len(G), dtype=bool),
               'only_F2_candidate': np.zeros(len(G), dtype=bool),
               'both_present_candidate': np.zeros(len(G), dtype=bool)}
        rd = np.zeros(len(G), dtype=np.float64)
        for u in range(p.n):
            a_, b_ = int(p.starts[u]), int(p.ends[u])
            # RAW units: buckets such as 3-5 refer to actual rank differences, so the
            # scaled training view must not be used here
            X = p.Xraw[u, :b_ - a_]
            src['only_F1_candidate'][a_:b_] = X[:, f['only_F1']] > 0.5
            src['only_F2_candidate'][a_:b_] = X[:, f['only_F2']] > 0.5
            src['both_present_candidate'][a_:b_] = X[:, f['both_present']] > 0.5
            rd[a_:b_] = X[:, f['abs_rank_difference']]
        for k, mask in src.items():
            v = G[mask]
            feat_rows.append([ds, f'source_{k}', int(len(v)),
                              f'{v.mean():.6f}' if len(v) else '', '', ''])
        for k, (lo, hi) in (('0-2', (0, 2)), ('3-5', (3, 5)), ('6-10', (6, 10)),
                            ('11-20', (11, 20)), ('>20', (21, 10 ** 9))):
            mask = (rd >= lo) & (rd <= hi)
            v = G[mask]
            feat_rows.append([ds, f'rankdis_{k}', int(len(v)),
                              f'{v.mean():.6f}' if len(v) else '', '', ''])
    # the feature-behaviour rows were collected for every dataset above; emit them with
    # their own writer (the gate_behaviour `with` block has already closed by now)
    with (out / 'test' / 'candidate_source_behaviour.csv').open('w', newline='') as fh:
        w2 = csv.writer(fh)
        w2.writerow(['dataset', 'view', 'n_candidates', 'mean_g', 'median_g', 'std_g'])
        for r in feat_rows:
            w2.writerow(r)

    # rescue vs damage
    with (out / 'test' / 'rank_rescue.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'n_uniform_failures', 'method', 'rescued@5', 'rescued@10',
                    'rescued@20'])
        for ds in DATASETS:
            d = detail[ds]; U = d['r_u']; ok = d['p'].eligible
            fail = ok & ((U < 0) | (U > 20))
            for meth, r in (('LSE', d['r_l']), ('Gate', d['r_g'])):
                rr = r[fail]
                w.writerow([ds, int(fail.sum()), meth,
                            int(((rr >= 1) & (rr <= 5)).sum()),
                            int(((rr >= 1) & (rr <= 10)).sum()),
                            int(((rr >= 1) & (rr <= 20)).sum())])
    with (out / 'test' / 'damage_analysis.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'n_uniform_successes', 'method', 'still_in_top20',
                    'pushed_out_top20', 'damage_rate'])
        for ds in DATASETS:
            d = detail[ds]; U = d['r_u']; ok = d['p'].eligible
            succ = ok & (U >= 1) & (U <= 20)
            for meth, r in (('LSE', d['r_l']), ('Gate', d['r_g'])):
                rr = r[succ]
                still = int(((rr >= 1) & (rr <= 20)).sum())
                n = int(succ.sum())
                w.writerow([ds, n, meth, still, n - still,
                            f'{(n - still) / n:.6f}' if n else ''])

    # efficiency (measured separately by dg_efficiency; placeholder if absent)
    man['test_evaluated'] = True
    man['test_evaluated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man['post_hoc_benchmark'] = True
    fp.write_text(json.dumps(man, indent=2, default=str))

    print('\n=== POST-HOC CANONICAL TEST BENCHMARK (pooled NDCG@20) ===')
    for m in methods:
        if any(raw[m][d] is None for d in DATASETS):
            print(f'  {m:14s} (reference only, see user_vs_candidate_routing.json)')
            continue
        r = np.concatenate([raw[m][d] for d in DATASETS])
        print(f'  {m:14s} {summarize(r)["NDCG@20"]:.6f}')
    print('\n=== per dataset Uniform / LSE / Gate ===')
    for ds in DATASETS:
        u = summarize(raw['Uniform'][ds])['NDCG@20']
        l = summarize(raw['LSE'][ds])['NDCG@20']
        g = summarize(raw['Gate'][ds])['NDCG@20']
        print(f'  {SHORT[ds]:12s} U={u:.6f} L={l:.6f} G={g:.6f}  '
              f'G-U={g-u:+.6f} G-L={g-l:+.6f}')
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
