#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE F (variance decomposition) + PHASE E/25-26 (unique-model NLL alignment)
+ PHASE J (predefined statistical comparisons).

All three read the same two artifacts produced earlier in this study:
  full_id_nll_all_factorizations.parquet   user x factorization target NLL
  pairwise_diversity_results.csv           2-model controlled study
  three_model_diversity_results.csv        3-model controlled study

Outputs
  unique_model_nll_alignment.csv
  unique_model_nll_matrix.csv
  variance_decomposition.csv
  statistical_tests.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import (  # noqa: E402
    DATASETS, FAMILY_LABELS, K_LIST, OUT, bootstrap_mean_ci,
    paired_bootstrap_chunked, two_sample_bootstrap_diff,
)

SEED = 20260912
N_PERM = 200


# ---------------------------------------------------------------------------
# unique-model NLL alignment
# ---------------------------------------------------------------------------
def build_wide(nll, ds):
    d = nll[nll['dataset'] == ds]
    col = d['family'] + '_k' + d['K'].astype(str)
    piv_nll = d.assign(col=col).pivot(index='user_index', columns='col',
                                      values='total_nll')
    piv_hit = d.assign(col=col).pivot(index='user_index', columns='col',
                                      values='hit20')
    return piv_nll, piv_hit


def alignment_for_set(ds, set_type, set_id, models, piv_nll, piv_hit, rows, mat_rows):
    nll = piv_nll[models].to_numpy(dtype=float)
    hit = piv_hit[models].to_numpy(dtype=int)
    U, M = nll.shape
    nhit = hit.sum(axis=1)
    argmin = nll.argmin(axis=1)
    uniq = nhit == 1
    uniq_idx = np.where(uniq)[0]
    own = hit.argmax(axis=1)[uniq_idx]
    match = (argmin[uniq_idx] == own)
    acc = float(match.mean()) if len(uniq_idx) else float('nan')

    global_model = int(nll.mean(axis=0).argmin())
    glob_acc = float((argmin[uniq_idx] == global_model).mean()) if len(uniq_idx) else float('nan')

    rng = np.random.default_rng(SEED)
    rand = rng.integers(0, M, size=(N_PERM, len(uniq_idx)))
    null_accs = (rand == argmin[uniq_idx][None, :]).mean(axis=1)
    p_perm = float((null_accs >= acc).mean()) if len(uniq_idx) else float('nan')

    for mi, m in enumerate(models):
        sel = uniq_idx[own == mi]
        if len(sel) == 0:
            continue
        rows.append({
            'row_type': 'group', 'dataset': ds, 'set_type': set_type,
            'set_id': set_id, 'n_models': M, 'model_set': '+'.join(models),
            'only_model': m, 'n_users': int(len(sel)),
            'mean_nll_at_own_model': float(nll[sel, mi].mean()),
            'mean_nll_at_other_models': float(
                np.delete(nll[sel], mi, axis=1).mean()),
            'argmin_match_accuracy': float((argmin[sel] == mi).mean()),
        })
        for mj, m2 in enumerate(models):
            mat_rows.append({'dataset': ds, 'set_type': set_type,
                             'set_id': set_id, 'only_model': m,
                             'nll_under_model': m2,
                             'mean_nll': float(nll[sel, mj].mean()),
                             'n_users': int(len(sel))})
    rows.append({
        'row_type': 'set_summary', 'dataset': ds, 'set_type': set_type,
        'set_id': set_id, 'n_models': M, 'model_set': '+'.join(models),
        'only_model': 'ALL', 'n_users': int(len(uniq_idx)),
        'argmin_match_accuracy': acc,
        'permutation_null_mean': float(null_accs.mean()) if len(uniq_idx) else float('nan'),
        'permutation_p_value': p_perm,
        'global_model_baseline': glob_acc,
        'global_model': models[global_model] if len(uniq_idx) else '',
        'random_baseline': 1.0 / M,
        'diagonal_minimum': None,
    })
    return acc


def unique_model_alignment(nll):
    rows, mat_rows = [], []
    for ds in DATASETS:
        piv_nll, piv_hit = build_wide(nll, ds)
        for f in FAMILY_LABELS:
            models = [f'{f}_k{k}' for k in K_LIST]
            if all(m in piv_nll.columns for m in models):
                alignment_for_set(ds, 'G', f, models, piv_nll, piv_hit, rows, mat_rows)
        for k in K_LIST:
            models = [f'{f}_k{k}' for f in FAMILY_LABELS]
            if all(m in piv_nll.columns for m in models):
                alignment_for_set(ds, 'P', f'K{k}', models, piv_nll, piv_hit,
                                  rows, mat_rows)
        # S-diversity: no teacher-forced NLL exists for the seed checkpoints
        rows.append({
            'row_type': 'not_available', 'dataset': ds, 'set_type': 'S',
            'set_id': 'seeds', 'n_models': None, 'model_set': '',
            'only_model': 'ALL', 'n_users': 0,
            'argmin_match_accuracy': None, 'permutation_null_mean': None,
            'permutation_p_value': None, 'global_model_baseline': None,
            'random_baseline': None, 'diagonal_minimum': None,
            'note': 'no teacher-forced NLL for seed checkpoints; S-axis NLL '
                    'alignment is NOT AVAILABLE',
        })
    out = pd.DataFrame(rows)
    mat = pd.DataFrame(mat_rows)
    # Diagonal check: for each Only-Model group, is the group's mean target NLL
    # under its OWN model the minimum across all models of that set?  (This is
    # the direct analogue of the re-audit's 16/16 Only-K diagonal result.)
    flags = []
    for (ds, st, sid), g in mat.groupby(['dataset', 'set_type', 'set_id']):
        piv = g.pivot(index='only_model', columns='nll_under_model',
                      values='mean_nll')
        for om in piv.index:
            row = piv.loc[om]
            flags.append({
                'dataset': ds, 'set_type': st, 'set_id': sid, 'only_model': om,
                'nll_row_min_model': str(row.idxmin()),
                'nll_at_own_model_mean': float(row[om]),
                'nll_row_min': float(row.min()),
                'nll_gap_to_row_min': float(row[om] - row.min()),
                'diagonal_minimum': bool(row.idxmin() == om),
            })
    fl = pd.DataFrame(flags)
    out = out.drop(columns=['diagonal_minimum'], errors='ignore').merge(
        fl, on=['dataset', 'set_type', 'set_id', 'only_model'], how='left')
    out.to_csv(OUT / 'unique_model_nll_alignment.csv', index=False)
    mat.to_csv(OUT / 'unique_model_nll_matrix.csv', index=False)
    print(f'wrote unique_model_nll_alignment.csv ({len(out)} rows)')
    print(f'  diagonal minima: {int(fl["diagonal_minimum"].sum())}/{len(fl)} groups')
    return out


# ---------------------------------------------------------------------------
# variance decomposition
# ---------------------------------------------------------------------------
def _share(ss):
    return {k: (v / ss['total']) for k, v in ss.items() if k != 'total'}


def variance_decomposition(nll, component='total_nll'):
    rows = []
    for ds in DATASETS:
        d = nll[nll['dataset'] == ds]
        fam_order = FAMILY_LABELS
        X = np.empty((d['user_index'].nunique(), len(fam_order), len(K_LIST)))
        users = np.sort(d['user_index'].unique())
        idx_u = {u: i for i, u in enumerate(users)}
        for fi, f in enumerate(fam_order):
            for ki, k in enumerate(K_LIST):
                sub = d[(d['family'] == f) & (d['K'] == k)].sort_values('user_index')
                assert np.array_equal(sub['user_index'].to_numpy(), users), \
                    'unbalanced design'
                X[:, fi, ki] = sub[component].to_numpy()
        U, Fa, Kk = X.shape
        grand = X.mean()
        sst = float(((X - grand) ** 2).sum())

        ss = {}
        ss['user'] = Kk * Fa * float(((X.mean(axis=(1, 2)) - grand) ** 2).sum())
        ss['family'] = U * Kk * float(((X.mean(axis=(0, 2)) - grand) ** 2).sum())
        ss['K'] = U * Fa * float(((X.mean(axis=(0, 1)) - grand) ** 2).sum())
        ss['family_x_K'] = U * float(((X.mean(axis=0) - grand) ** 2).sum()) \
            - ss['family'] - ss['K']
        ss['user_x_family'] = Kk * float(((X.mean(axis=2) - grand) ** 2).sum()) \
            - ss['user'] - ss['family']
        ss['user_x_K'] = Fa * float(((X.mean(axis=1) - grand) ** 2).sum()) \
            - ss['user'] - ss['K']
        ss['user_x_family_x_K'] = sst - (ss['user'] + ss['family'] + ss['K']
                                         + ss['family_x_K'] + ss['user_x_family']
                                         + ss['user_x_K'])

        # two-way user x (family,K) decomposition, 12 factorizations
        X2 = X.reshape(U, Fa * Kk)
        g2 = X2.mean()
        sst2 = float(((X2 - g2) ** 2).sum())
        ss2 = {
            'user': Fa * Kk * float(((X2.mean(axis=1) - g2) ** 2).sum()),
            'factorization': U * float(((X2.mean(axis=0) - g2) ** 2).sum()),
        }
        ss2['user_x_factorization'] = sst2 - ss2['user'] - ss2['factorization']

        def emit(model, s, tot):
            for name, val in s.items():
                rows.append({
                    'dataset': ds, 'component': component, 'model': model,
                    'effect': name, 'sum_of_squares': val,
                    'variance_share': val / tot, 'total_ss': tot,
                    'n_users': U, 'n_family': Fa, 'n_K': Kk,
                })
            rows.append({'dataset': ds, 'component': component, 'model': model,
                         'effect': 'TOTAL', 'sum_of_squares': tot,
                         'variance_share': 1.0, 'total_ss': tot,
                         'n_users': U, 'n_family': Fa, 'n_K': Kk})

        emit('three_way_user_family_K', ss, sst)
        emit('two_way_user_factorization', ss2, sst2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# statistical tests
# ---------------------------------------------------------------------------
def dist_stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    ci = bootstrap_mean_ci(v, seed=SEED)
    return {'n': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'std': float(v.std(ddof=1)) if v.size > 1 else float('nan'),
            'min': float(v.min()), 'max': float(v.max()),
            'ci_lo': ci['ci_lo'], 'ci_hi': ci['ci_hi']}


def add_test(rows, test, ds, metric, a, b, label_a, label_b, note=''):
    sa, sb = dist_stats(a), dist_stats(b)
    if not sa or not sb:
        return
    small = min(sa['n'], sb['n']) < 3
    diff = two_sample_bootstrap_diff(a, b, seed=SEED)
    if small:
        # a bootstrap over 1-2 pairs is degenerate: its CI collapses and its
        # p-value is an artefact.  Report the direction, refuse the inference.
        diff = {'diff': diff['diff'], 'ci_lo': None, 'ci_hi': None,
                'p_value': None}
        note = (note + ' | n<3 in one group: bootstrap not meaningful, '
                        'direction only').strip(' |')
    rows.append({
        'test': test, 'dataset': ds, 'metric': metric,
        'group_a': label_a, 'group_b': label_b,
        'n_a': sa['n'], 'n_b': sb['n'],
        'mean_a': sa['mean'], 'mean_b': sb['mean'],
        'median_a': sa['median'], 'median_b': sb['median'],
        'std_a': sa['std'], 'std_b': sb['std'],
        'min_a': sa['min'], 'max_a': sa['max'],
        'min_b': sb['min'], 'max_b': sb['max'],
        'mean_a_ci_lo': sa['ci_lo'], 'mean_a_ci_hi': sa['ci_hi'],
        'mean_b_ci_lo': sb['ci_lo'], 'mean_b_ci_hi': sb['ci_hi'],
        'diff_a_minus_b': diff['diff'], 'diff_ci_lo': diff['ci_lo'],
        'diff_ci_hi': diff['ci_hi'], 'diff_p_value': diff['p_value'],
        'significant_95': (None if small else int(diff.get('significant_95', 0))),
        'method': ('direction only (n<3)' if small else
                   'two-sample bootstrap over MODEL PAIRS (10000 resamples)'),
        'note': note,
    })


def statistical_tests(pw, tr, align):
    rows = []
    METRICS_2 = ['fusion_gain', 'oracle_gain', 'oracle_exploited_fraction',
                 'hit_disagreement', 'candidate_jaccard', 'fusion_ndcg20']
    for ds in DATASETS:
        g = pw[(pw['dataset'] == ds) & (pw['diversity_type'] == 'G')]
        p = pw[(pw['dataset'] == ds) & (pw['diversity_type'] == 'P')]
        for m in METRICS_2:
            add_test(rows, 'Test1_2model_G_vs_P', ds, m,
                     g[m].to_numpy(), p[m].to_numpy(), 'G(same family,diff K)',
                     'P(same K,diff family)',
                     'pair-level distributions; pair sets are not naturally matched')
    METRICS_3 = ['fusion_gain', 'oracle_gain', 'oracle_exploited_fraction',
                 'mean_hit_disagreement', 'fusion_ndcg20']
    for ds in DATASETS:
        g = tr[(tr['dataset'] == ds) & (tr['diversity_type'] == 'G')]
        p = tr[(tr['dataset'] == ds) & (tr['diversity_type'] == 'P')]
        for m in METRICS_3:
            add_test(rows, 'Test2_3model_G_vs_P', ds, m,
                     g[m].to_numpy(), p[m].to_numpy(), 'G-triplets', 'P-triplets',
                     'all G-triplets reported, never only the test-best one')

    # Test 3: factorization diversity vs seed diversity, matched on dataset and K
    s = pw[pw['diversity_type'] == 'S']
    for ds in DATASETS:
        for k in K_LIST:
            sd = s[(s['dataset'] == ds) & (s['K_A'] == k)]
            if len(sd) == 0:
                continue
            for m in ['hit_disagreement', 'candidate_jaccard', 'fusion_gain',
                      'oracle_gain', 'fusion_ndcg20']:
                pd_ = pw[(pw['dataset'] == ds) & (pw['diversity_type'] == 'P') &
                         (pw['K_A'] == k)]
                gd = pw[(pw['dataset'] == ds) & (pw['diversity_type'] == 'G')]
                add_test(rows, 'Test3_factorization_vs_seed', f'{ds}|K{k}', m,
                         pd_[m].to_numpy(), sd[m].to_numpy(),
                         f'P(same K{k},diff family)', f'S(same family+K{k},diff seed)',
                         'matched on dataset and K; small n')
                add_test(rows, 'Test3_factorization_vs_seed', f'{ds}|K{k}', m,
                         gd[m].to_numpy(), sd[m].to_numpy(),
                         'G(same family,diff K)', f'S(same family+K{k},diff seed)',
                         'matched on dataset only; small n')

    # Test 3b: same comparison restricted to QUALITY-MATCHED seed pairs, so the
    # result cannot be produced by seed checkpoints that never converged.
    sens_path = OUT / 'seed_diversity_sensitivity.csv'
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        q = sens[sens['quality_matched']]
        for (ds, k), sg in q.groupby(['dataset', 'K_A']):
            for m in ['hit_disagreement', 'candidate_jaccard', 'fusion_gain',
                      'oracle_gain', 'fusion_ndcg20']:
                if m not in sg.columns:
                    continue
                pd_ = pw[(pw['dataset'] == ds) & (pw['diversity_type'] == 'P') &
                         (pw['K_A'] == k)]
                add_test(rows, 'Test3b_factorization_vs_seed_quality_matched',
                         f'{ds}|K{k}', m, pd_[m].to_numpy(), sg[m].to_numpy(),
                         f'P(same K{k},diff family)',
                         f'S quality-matched (same family+K{k},diff seed)',
                         'seed checkpoints below 0.5x canonical NDCG@20 excluded; '
                         'fixed pre-declared convergence filter')

    # alignment accuracy vs its own baselines
    al = align[align['row_type'] == 'set_summary']
    for _, r in al.iterrows():
        rows.append({
            'test': 'Test4_own_factorization_NLL_alignment',
            'dataset': r['dataset'], 'metric': 'argmin_match_accuracy',
            'group_a': f"{r['set_type']}:{r['set_id']}", 'group_b': 'random_baseline',
            'n_a': r['n_users'], 'n_b': r['n_models'],
            'mean_a': r['argmin_match_accuracy'], 'mean_b': r['random_baseline'],
            'median_a': None, 'median_b': None, 'std_a': None, 'std_b': None,
            'min_a': None, 'max_a': None, 'min_b': None, 'max_b': None,
            'mean_a_ci_lo': None, 'mean_a_ci_hi': None,
            'mean_b_ci_lo': None, 'mean_b_ci_hi': None,
            'diff_a_minus_b': (None if r['argmin_match_accuracy'] is None
                               else r['argmin_match_accuracy'] - r['random_baseline']),
            'diff_ci_lo': None, 'diff_ci_hi': None, 'diff_p_value': None,
            'significant_95': None,
            'method': 'descriptive vs analytic null',
            'note': (f"permutation null {r['permutation_null_mean']:.4f}, "
                     f"global-model baseline {r['global_model_baseline']:.4f}, "
                     f"p_perm {r['permutation_p_value']}"
                     if r['permutation_null_mean'] is not None and
                     np.isfinite(r['permutation_null_mean']) else ''),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'statistical_tests.csv', index=False)
    print(f'wrote statistical_tests.csv ({len(out)} rows)')
    return out


def main():
    nll = pd.read_parquet(OUT / 'full_id_nll_all_factorizations.parquet')
    print(f'loaded full_id_nll_all_factorizations.parquet {nll.shape}')
    align = unique_model_alignment(nll)
    parts = [variance_decomposition(nll, c)
             for c in ('total_nll', 'semantic_nll', 'collision_nll')]
    vd = pd.concat(parts, ignore_index=True)
    vd[vd['component'] == 'total_nll'].to_csv(OUT / 'variance_decomposition.csv',
                                              index=False)
    for comp in ('semantic_nll', 'collision_nll'):
        vd[vd['component'] == comp].to_csv(
            OUT / f'variance_decomposition_{comp}.csv', index=False)
    print(f'wrote variance_decomposition.csv ({len(vd[vd["component"]=="total_nll"])} rows, '
          f'component=total_nll); semantic/collision components in side files')

    pw = pd.read_csv(OUT / 'pairwise_diversity_results.csv')
    tr_path = OUT / 'three_model_diversity_results.csv'
    tr = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
    if len(tr):
        statistical_tests(pw, tr, align)
    else:
        print('three_model_diversity_results.csv missing; Test2/Test3 skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
