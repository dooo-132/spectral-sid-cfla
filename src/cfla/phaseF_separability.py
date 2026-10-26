#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE F -- Is the cross-K difference a change of *factorization* or a change of
*overall generation difficulty*?

Consumes outputs/mechanism/full_id_reaudit/user_level_full_id_metrics.parquet
(written by analysis/mechanism/full_id_reaudit.py; no training, no new model).

The discriminating question
---------------------------
"K changes overall generation difficulty" predicts:
    (i)  a large *main effect of K* on the teacher-forced NLL of the target SID,
    (ii) that main effect being the same sign and roughly the same size for every
         user (a uniform shift),
    (iii) the per-user argmin over K being the same K for everybody
          (namely the globally easiest one).
"K changes how item identity is *discretised* (factorised)" predicts:
    (iv) a small main effect of K, and
    (v)  a large user x K interaction -- the ordering of K differs from user to
         user,
    (vi) that per-user ordering being *predictive* of which K alone can retrieve
         that user's target.

Outputs -> outputs/mechanism/full_id_reaudit/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

IN = REPO / 'outputs' / 'mechanism' / 'full_id_reaudit'
OUT = IN
DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
SHORT = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'CellPhones',
         'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}
KS = [2, 3, 4, 5]
NBOOT = 2000
SEED = 20260912


# --------------------------------------------------------------------------- #
def success_set_table(u, metric):
    """Hit-set category x K matrix of mean NLL, and the diagonal check."""
    rows, diag = [], []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        wide = s.pivot_table(index='user_index', columns='K', values=metric)
        hit = s.pivot_table(index='user_index', columns='K', values='hit20')
        wide, hit = wide[KS], hit[KS]
        cat = hit.apply(lambda r: tuple(int(x) for x in r.values), axis=1)
        names = {}
        for k in KS:
            names[tuple(1 if j == k else 0 for j in KS)] = f'Only K{k}'
        names[tuple([1] * 4)] = 'All K'
        names[tuple([1, 1, 1, 0])] = 'Multiple K'
        for c, g in cat.groupby(cat):
            lab = names.get(c)
            if lab is None:
                lab = 'Multiple K' if sum(c) > 1 else 'No K'
            idx = g.index
            row = {'dataset': ds, 'category': lab, 'n_users': len(idx),
                   'hit_pattern': ''.join(str(x) for x in c)}
            for k in KS:
                row[f'NLL_K{k}'] = float(wide.loc[idx, k].mean())
            rows.append(row)
            if lab.startswith('Only K'):
                k0 = int(lab.split('K')[1])
                others = [wide.loc[idx, k].mean() for k in KS if k != k0]
                diag.append({'dataset': ds, 'only_K': k0, 'n_users': len(idx),
                             f'NLL_onlyK{k0}': float(wide.loc[idx, k0].mean()),
                             'mean_NLL_other_K': float(np.mean(others)),
                             'is_minimum': bool(wide.loc[idx, k0].mean()
                                                <= min(others) + 1e-12)})
    return pd.DataFrame(rows), pd.DataFrame(diag)


def onlyk_paired(u, metric, nboot=NBOOT, seed=SEED):
    """Within each Only-K group, paired user-level difference NLL(onlyK) - NLL(K')
    with a bootstrap CI and a sign test."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        wide = s.pivot_table(index='user_index', columns='K', values=metric)[KS]
        hit = s.pivot_table(index='user_index', columns='K', values='hit20')[KS]
        for k0 in KS:
            m = (hit[k0] == 1) & (hit[[k for k in KS if k != k0]].sum(axis=1) == 0)
            idx = m[m].index
            if len(idx) < 20:
                continue
            for k2 in KS:
                if k2 == k0:
                    continue
                d = (wide.loc[idx, k0] - wide.loc[idx, k2]).to_numpy()
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                               for _ in range(nboot)])
                rows.append({
                    'dataset': ds, 'metric': metric, 'only_K': k0, 'vs_K': k2,
                    'n_users': len(d), 'mean_diff': float(d.mean()),
                    'ci_lo': float(np.percentile(bs, 2.5)),
                    'ci_hi': float(np.percentile(bs, 97.5)),
                    'frac_negative': float((d < 0).mean()),
                    'sign_test_p': float(_binom_two_sided((d < 0).sum(), len(d))),
                })
    return pd.DataFrame(rows)


def _binom_two_sided(k, n):
    from math import comb
    if n == 0:
        return float('nan')
    p = sum(comb(n, i) for i in range(0, n + 1) if
            abs(i - n / 2) >= abs(k - n / 2)) / (2 ** n)
    return min(1.0, p)


def variance_decomposition(u, metric):
    """Two-way (user x K) variance decomposition of the NLL table.

    Balanced by construction: every user is present for all four K.
    SS_total = SS_user + SS_K + SS_resid, with
        df_user   = nU - 1
        df_K      = nK - 1
        df_resid  = (nU - 1)(nK - 1)
    Reported as the fraction of total sum of squares.
    """
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        w = s.pivot_table(index='user_index', columns='K', values=metric)[KS]
        X = w.to_numpy()
        nU, nK = X.shape
        gm = X.mean()
        ss_tot = ((X - gm) ** 2).sum()
        ss_user = nK * ((X.mean(axis=1) - gm) ** 2).sum()
        ss_K = nU * ((X.mean(axis=0) - gm) ** 2).sum()
        ss_res = ss_tot - ss_user - ss_K
        ms_res = ss_res / ((nU - 1) * (nK - 1))
        f_K = (ss_K / (nK - 1)) / ms_res
        rows.append({
            'dataset': ds, 'metric': metric, 'n_users': nU, 'n_K': nK,
            'ss_total': float(ss_tot),
            'frac_ss_user': float(ss_user / ss_tot),
            'frac_ss_K_main': float(ss_K / ss_tot),
            'frac_ss_interaction': float(ss_res / ss_tot),
            'F_K_main': float(f_K),
            'sd_of_K_means': float(X.mean(axis=0).std(ddof=0)),
            'sd_of_user_means': float(X.mean(axis=1).std(ddof=0)),
            'mean_within_user_sd': float(X.std(axis=1, ddof=0).mean()),
        })
    return pd.DataFrame(rows)


def argmin_analysis(u):
    """Per-user argmin over K of the NLL, vs which K hits."""
    rows, agree = [], []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        for metric in ['total_nll', 'semantic_nll', 'collision_nll']:
            w = s.pivot_table(index='user_index', columns='K', values=metric)[KS]
            h = s.pivot_table(index='user_index', columns='K', values='hit20')[KS]
            am = w.to_numpy().argmin(axis=1)
            dist = {f'argmin_K{k}': float((am == i).mean())
                    for i, k in enumerate(KS)}
            # for users with a unique hit, does argmin land on it?
            uniq = h.sum(axis=1) == 1
            if uniq.sum() > 0:
                onlyk = h.to_numpy()[uniq.to_numpy()].argmax(axis=1)
                acc = float((am[uniq.to_numpy()] == onlyk).mean())
            else:
                acc = float('nan')
            # chance for landing on the unique-hit K == 1/4
            rows.append({'dataset': ds, 'metric': metric,
                         'n_users': len(w), 'n_users_unique_hit': int(uniq.sum()),
                         'P_argmin_matches_unique_hit_K': acc,
                         'chance': 0.25, **dist})
            # agreement between metric-specific argmin and total-NLL argmin
            agree.append({'dataset': ds, 'metric': metric,
                          'argmin_agreement_with_total_nll': float('nan')})
    return pd.DataFrame(rows), pd.DataFrame(agree)


def permutation_null(u, metric='total_nll', nperm=200, seed=SEED):
    """Empirical null for P(argmin_K NLL == the unique-hit K).

    Within each user the *number* of hits is preserved; the labels of which K
    hits are permuted independently per user. The NLL table is held fixed. This
    null therefore still gives "easy users" every advantage they have through
    their NLL level -- it removes only the alignment between the NLL ordering
    across K and the retrieval outcome. If the observed statistic survives this
    null, the cross-K ordering carries the signal, not the user's overall level.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        w = s.pivot_table(index='user_index', columns='K', values=metric)[KS]
        h = s.pivot_table(index='user_index', columns='K', values='hit20')[KS]
        am = w.to_numpy().argmin(axis=1)
        H = h.to_numpy().astype(np.int8)
        uniq = H.sum(axis=1) == 1
        obs = float((am[uniq] == H[uniq].argmax(axis=1)).mean())
        m = int(uniq.sum())
        null = np.empty(nperm)
        idx = np.where(uniq)[0]
        for p in range(nperm):
            Hp = np.zeros_like(H)
            Hp[idx, rng.integers(0, 4, len(idx))] = 1
            null[p] = (am[idx] == Hp[idx].argmax(axis=1)).mean()
        # global-level control: same statistic but argmin replaced by the
        # globally easiest K (constant), which is what "user level only" predicts
        gm = w.mean(axis=0).to_numpy().argmin()
        rows.append({'dataset': ds, 'metric': metric,
                     'n_users_unique_hit': m,
                     'observed_P_argmin_matches_unique_hit': obs,
                     'perm_null_mean': float(null.mean()),
                     'perm_null_p97.5': float(np.percentile(null, 97.5)),
                     'p_value': float((null >= obs).mean()),
                     'global_K_baseline': float((gm == H[idx].argmax(axis=1)).mean()),
                     'global_K': KS[gm]})
    return pd.DataFrame(rows)


def composition_standardization():
    """Direct standardisation: split the crude cross-K trend of a stratified rate
    into a *composition* part (the SID mapping moves users between collision-group
    buckets) and a *rate* part (the per-bucket rate itself changes).

    crude_K   = sum_b w_bK * r_bK
    std_K     = sum_b w_b(pooled) * r_bK      <- common mix, K-specific rates
    comp_K    = crude_K - std_K               <- contribution of the mix shift
    """
    d = pd.read_csv(IN / 'collision_group_stratified_metrics.csv')
    rows = []
    for ds in DATASETS:
        s = d[d.dataset == ds]
        w = s.pivot_table(index='group_bucket', columns='K', values='n_users')
        w = w / w.sum()
        pooled = w.mean(axis=1)
        pooled = pooled / pooled.sum()
        for metric in ['collision_token_accuracy', 'item_hit@20',
                       'full_sid_top1_accuracy']:
            r = s.pivot_table(index='group_bucket', columns='K', values=metric)[KS]
            crude = (r * w).sum()
            std = (r * pooled.values[:, None]).sum()
            rows.append({
                'dataset': ds, 'metric': metric,
                **{f'crude_K{k}': float(crude[k]) for k in KS},
                **{f'std_K{k}': float(std[k]) for k in KS},
                **{f'composition_K{k}': float(crude[k] - std[k]) for k in KS},
                'crude_trend_K5_minus_K2': float(crude[5] - crude[2]),
                'std_trend_K5_minus_K2': float(std[5] - std[2]),
                'composition_share_of_trend': float(
                    (crude[5] - crude[2] - (std[5] - std[2]))
                    / (crude[5] - crude[2])) if abs(crude[5] - crude[2]) > 1e-12
                else float('nan'),
            })
    return pd.DataFrame(rows)


def structural_entropy_accounting():
    """Total conditional entropy of the item identity under the tokenizer.

    If K re-factorised the identity without changing how much information is
    needed to name it, the SUM over positions of H(C_t | C_<t) is invariant in K,
    while its split between semantic slots and the disambiguation slot is not.
    The model's teacher-forced cross-entropy is compared against it.
    """
    s = pd.read_csv(IN / 'structural_vs_model_position_cost.csv')
    tot = (s.groupby(['dataset', 'K'])
             .agg(H=('structural_conditional_entropy', 'sum'),
                  CE=('model_cross_entropy', 'sum'))
             .reset_index())
    tot['CE_minus_H'] = tot.CE - tot.H
    tot['CE_over_H'] = tot.CE / tot.H
    tot['H_semantic'] = [
        float(s[(s.dataset == r.dataset) & (s.K == r.K)
                & (s.token_type == 'semantic')].structural_conditional_entropy.sum())
        for r in tot.itertuples()]
    tot['H_collision'] = [
        float(s[(s.dataset == r.dataset) & (s.K == r.K)
                & (s.token_type == 'collision')].structural_conditional_entropy.sum())
        for r in tot.itertuples()]
    tot['H_spread_within_dataset'] = tot.groupby('dataset').H.transform(
        lambda x: x.max() - x.min())
    return tot


def rank_agreement(u):
    """Per-user Spearman between the K-ordering implied by NLL and by item rank."""
    from scipy.stats import spearmanr
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        nll = s.pivot_table(index='user_index', columns='K', values='total_nll')[KS]
        rk = s.pivot_table(index='user_index', columns='K', values='rank')[KS]
        hit = s.pivot_table(index='user_index', columns='K', values='hit20')[KS]
        m = hit.sum(axis=1) >= 1
        rho = [spearmanr(nll.loc[i].to_numpy(), -rk.loc[i].to_numpy()).statistic
               for i in m[m].index]
        rho = np.array([r for r in rho if np.isfinite(r)])
        bs = _boot_mean(rho, NBOOT, SEED)
        rows.append({'dataset': ds, 'n_users_with_any_hit': int(m.sum()),
                     'mean_spearman_NLL_vs_negrank': float(rho.mean()),
                     'ci_lo': float(np.percentile(bs, 2.5)),
                     'ci_hi': float(np.percentile(bs, 97.5)),
                     'frac_positive': float((rho > 0).mean())})
    return pd.DataFrame(rows)


def _boot_mean(x, nboot, seed):
    rng = np.random.default_rng(seed)
    return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(nboot)])


def converse_table(u):
    """For users whose per-user NLL argmin is K, how often is K the unique hit?"""
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        w = s.pivot_table(index='user_index', columns='K', values='total_nll')[KS]
        h = s.pivot_table(index='user_index', columns='K', values='hit20')[KS]
        am = pd.Series(w.to_numpy().argmin(axis=1), index=w.index)
        for i, k in enumerate(KS):
            m = am == i
            if m.sum() == 0:
                continue
            hh = h.loc[m[m].index]
            rows.append({'dataset': ds, 'argmin_K': k, 'n_users': int(m.sum()),
                         'P_K_hits': float(hh[k].mean()),
                         'P_K_is_the_unique_hit': float(
                             ((hh[k] == 1) & (hh[[x for x in KS if x != k]]
                                              .sum(axis=1) == 0)).mean()),
                         'P_no_K_hits': float((hh.sum(axis=1) == 0).mean())})
    return pd.DataFrame(rows)


def global_vs_user_ordering(u):
    """How much does the per-user K-ordering deviate from the global K-ordering?"""
    rows = []
    for ds in DATASETS:
        s = u[u.dataset == ds]
        w = s.pivot_table(index='user_index', columns='K', values='total_nll')[KS]
        gm = w.mean(axis=0).to_numpy()
        gorder = np.argsort(np.argsort(gm))
        X = w.to_numpy()
        uorder = np.argsort(np.argsort(X, axis=1), axis=1)
        exact = float((uorder == gorder[None, :]).all(axis=1).mean())
        rows.append({'dataset': ds,
                     'global_argmin_K': KS[int(gm.argmin())],
                     'global_K_order_easiest_to_hardest':
                         ' < '.join(str(KS[i]) for i in np.argsort(gm)),
                     'P_user_order_equals_global_order': exact,
                     'P_user_argmin_equals_global_argmin':
                         float((X.argmin(axis=1) == gm.argmin()).mean()),
                     'chance_argmin_equals_global': 0.25})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    u = pd.read_parquet(IN / 'user_level_full_id_metrics.parquet')
    assert set(u.dataset.unique()) == set(DATASETS)
    assert sorted(u.K.unique()) == KS
    bal = u.groupby(['dataset', 'K']).size().unstack()
    assert (bal.nunique(axis=1) == 1).all(), 'table is not balanced across K'
    print('balanced user x K table confirmed')

    cat_tab, diag = success_set_table(u, 'total_nll')
    cat_tab.to_csv(OUT / 'phaseF_category_nll_matrix.csv', index=False)
    diag.to_csv(OUT / 'phaseF_onlyK_diagonal.csv', index=False)
    print('diagonal minimum holds for',
          f'{int(diag.is_minimum.sum())}/{len(diag)} Only-K cells')

    pd.concat([onlyk_paired(u, 'total_nll'),
               onlyk_paired(u, 'semantic_nll')],
              ignore_index=True).to_csv(
        OUT / 'phaseF_onlyK_paired_nll_tests.csv', index=False)

    var = pd.concat([variance_decomposition(u, m)
                     for m in ['total_nll', 'semantic_nll', 'collision_nll']],
                    ignore_index=True)
    var.to_csv(OUT / 'phaseF_variance_decomposition.csv', index=False)

    am, _ = argmin_analysis(u)
    am.to_csv(OUT / 'phaseF_argmin_vs_hit.csv', index=False)

    rank_agreement(u).to_csv(OUT / 'phaseF_nll_vs_rank_agreement.csv', index=False)
    perm = pd.concat([permutation_null(u, m)
                      for m in ['total_nll', 'semantic_nll', 'collision_nll']],
                     ignore_index=True)
    perm.to_csv(OUT / 'phaseF_permutation_null.csv', index=False)
    print(perm.to_string(index=False))
    converse_table(u).to_csv(OUT / 'phaseF_converse_argmin_K.csv', index=False)
    glob = global_vs_user_ordering(u)
    glob.to_csv(OUT / 'phaseF_global_vs_user_ordering.csv', index=False)

    comp = composition_standardization()
    comp.to_csv(OUT / 'phaseF_composition_standardization.csv', index=False)
    ent = structural_entropy_accounting()
    ent.to_csv(OUT / 'phaseF_total_structural_entropy.csv', index=False)
    print(ent.round(6).to_string(index=False))
    print('H spread within dataset (must be ~0):',
          ent.H_spread_within_dataset.unique())

    print(glob.to_string(index=False))
    print(var[var.metric == 'total_nll'].to_string(index=False))
    print(diag.to_string(index=False))

    src = IN / 'user_level_full_id_metrics.parquet'
    prov = {
        'phase': 'phaseF_separability',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': ('does K change how item identity is factorised, or overall '
                     'generation difficulty?'),
        'inputs': {'user_level_full_id_metrics.parquet':
                   {'sha256': _sha256(src)}},
        'training_performed': 0,
        'bootstrap_resamples': NBOOT, 'bootstrap_seed': SEED,
        'permutation_draws': 200, 'permutation_seed': SEED,
        'n_users_total': int(len(u)),
        'datasets': DATASETS, 'K': KS,
    }
    (OUT / 'phaseF_PROVENANCE.json').write_text(json.dumps(prov, indent=2))
    print(f'\nPhase F done in {time.time()-t0:.1f}s')


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    main()
