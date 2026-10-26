#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis figures for the factorization-generality study.

Names are analysis-scoped on purpose (`analysis_*.png`); nothing here is a
camera-ready or paper figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import DATASETS, FIGDIR, OUT  # noqa: E402

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': 0.25})
DTYPES = ['G', 'P', 'S']
DTITLE = {'G': 'G: same family, different K',
          'P': 'P: same K, different family',
          'S': 'S: same family+K, different seed'}
COLORS = {'G': '#1f77b4', 'P': '#d62728', 'S': '#2ca02c'}


def box_by_diversity(df, value, ylabel, fname, title):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    for ax, ds in zip(axes.ravel(), DATASETS):
        g = df[df['dataset'] == ds]
        present = [d for d in DTYPES if (g['diversity_type'] == d).any()]
        data = [g.loc[g['diversity_type'] == d, value].dropna().to_numpy() for d in present]
        bp = ax.boxplot(data, labels=[DTITLE[d].split(':')[0] for d in present],
                        widths=0.5, showfliers=False, patch_artist=True)
        for patch, d in zip(bp['boxes'], present):
            patch.set_facecolor(COLORS[d]); patch.set_alpha(0.35)
        for i, (d, arr) in enumerate(zip(present, data), start=1):
            x = np.full(len(arr), i) + np.random.default_rng(0).uniform(-0.11, 0.11, len(arr))
            ax.scatter(x, arr, s=9, color=COLORS[d], zorder=3, alpha=0.8)
        ax.set_title(ds.replace('_', ' '), fontsize=9)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches='tight')
    plt.close(fig)
    print('wrote', fname)


def fig_distance_vs_disagreement(fd, pw):
    key = ['dataset', 'family_A', 'family_B', 'K_A', 'K_B']
    can = pw[(pw['seed_A'] == 2025) & (pw['seed_B'] == 2025)]
    fd = fd.drop(columns=[c for c in ('diversity_type', 'same_family', 'same_K')
                          if c in fd.columns])
    m = can.merge(fd, on=key, how='inner')
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for ax, ds in zip(axes.ravel(), DATASETS):
        g = m[m['dataset'] == ds]
        for d, sub in g.groupby('diversity_type'):
            ax.scatter(sub['base_prefix_ari'], sub['hit_disagreement'],
                       s=26, alpha=0.85, color=COLORS.get(d, 'gray'),
                       label=DTITLE.get(d, d))
        if len(g) >= 4:
            r = np.corrcoef(g['base_prefix_ari'], g['hit_disagreement'])[0, 1]
            rho = pd.Series(g['base_prefix_ari']).corr(
                pd.Series(g['hit_disagreement']), method='spearman')
            ax.set_title(f'{ds.replace("_", " ")}\nr={r:.2f}  rho={rho:.2f}  n={len(g)}',
                         fontsize=9)
        ax.set_xlabel('base-prefix partition ARI (structural similarity)')
        ax.set_ylabel('Hit@20 disagreement')
    axes[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('Structural factorization distance vs retrieval disagreement '
                 '(descriptive; small n)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'analysis_factorization_distance_vs_hit_disagreement.png',
                bbox_inches='tight')
    plt.close(fig)
    print('wrote analysis_factorization_distance_vs_hit_disagreement.png')


def fig_nll_alignment(al):
    g = al[al['row_type'] == 'set_summary'].copy()
    g = g[g['set_type'].isin(['G', 'P'])]
    g['tag'] = g['dataset'].str.replace('_', ' ').str.slice(0, 12) + '\n' + \
        g['set_type'] + ':' + g['set_id'].astype(str)
    g = g.sort_values(['dataset', 'set_type', 'set_id'])
    fig, ax = plt.subplots(figsize=(max(9, 0.30 * len(g)), 4.6))
    x = np.arange(len(g))
    ax.bar(x, g['argmin_match_accuracy'], color='#4c72b0', label='argmin == unique-hit model')
    for i, (_, r) in enumerate(g.iterrows()):
        ax.hlines(r['random_baseline'], i - 0.4, i + 0.4, color='crimson',
                  linestyle='--', linewidth=1.1)
        if r['global_model_baseline'] is not None and np.isfinite(r['global_model_baseline']):
            ax.plot(i, r['global_model_baseline'], marker='x', color='black', ms=5)
    ax.axhline(0.25, color='crimson', linestyle='--', linewidth=1.1,
               label='random baseline 1/M (M=4); per-set 1/M drawn')
    ax.plot([], [], marker='x', color='black', linestyle='none',
            label='always guess globally cheapest model')
    ax.set_xticks(x); ax.set_xticklabels(g['tag'], rotation=90, fontsize=6.5)
    ax.set_ylabel('P(argmin model == the only hitting model)')
    ax.set_ylim(0, 1.02)
    ax.set_title('Own-factorization target-NLL alignment on Only-Model users')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'analysis_nll_alignment_accuracy.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote analysis_nll_alignment_accuracy.png')


def fig_variance(vd):
    v = vd[(vd['model'] == 'three_way_user_family_K')]
    effects = ['user', 'family', 'K', 'family_x_K', 'user_x_family', 'user_x_K',
               'user_x_family_x_K']
    labels = ['user', 'family', 'K', 'family×K', 'user×family', 'user×K',
              'user×family×K']
    piv = v.pivot_table(index='dataset', columns='effect', values='variance_share')
    piv = piv.reindex(columns=effects, fill_value=0.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bottom = np.zeros(len(piv))
    cmap = plt.get_cmap('tab20')
    for i, (e, lab) in enumerate(zip(effects, labels)):
        vals = piv[e].to_numpy()
        ax.bar(np.arange(len(piv)), vals, bottom=bottom, label=lab,
               color=cmap(i % 20))
        for j, (b, val) in enumerate(zip(bottom, vals)):
            if val > 0.03:
                ax.text(j, b + val / 2, f'{100*val:.1f}%', ha='center',
                        va='center', fontsize=7)
        bottom += vals
    ax.set_xticks(np.arange(len(piv)))
    ax.set_xticklabels([d.replace('_', ' ') for d in piv.index], rotation=15, fontsize=8)
    ax.set_ylabel('share of variance in teacher-forced TotalNLL')
    ax.set_ylim(0, 1.0)
    ax.set_title('User x factorization variance decomposition (family x K)')
    ax.legend(fontsize=7, ncol=2, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIGDIR / 'analysis_variance_decomposition.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote analysis_variance_decomposition.png')


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pw = pd.read_csv(OUT / 'pairwise_diversity_results.csv')
    box_by_diversity(pw, 'fusion_gain', 'NDCG@20 fusion gain over stronger single',
                     'analysis_fusion_gain_by_diversity_type.png',
                     'Two-model fusion gain by diversity type')
    box_by_diversity(pw, 'oracle_gain', 'NDCG@20 oracle gain over stronger single',
                     'analysis_oracle_gap_by_diversity_type.png',
                     'Two-model oracle gap by diversity type')
    box_by_diversity(pw, 'hit_disagreement', 'P(Hit@20 disagreement)',
                     'analysis_hit_disagreement_by_diversity_type.png',
                     'Two-model Hit@20 disagreement by diversity type')

    fdp = OUT / 'factorization_distance.csv'
    if fdp.exists():
        fig_distance_vs_disagreement(pd.read_csv(fdp), pw)
    ap = OUT / 'unique_model_nll_alignment.csv'
    if ap.exists():
        fig_nll_alignment(pd.read_csv(ap))
    vp = OUT / 'variance_decomposition.csv'
    if vp.exists():
        fig_variance(pd.read_csv(vp))
    return 0


if __name__ == '__main__':
    sys.exit(main())
