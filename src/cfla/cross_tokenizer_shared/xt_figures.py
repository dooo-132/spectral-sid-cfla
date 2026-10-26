#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis figures for the cross-tokenizer / shared-parameter study.

All figures are analysis-scoped; nothing here is a paper figure.
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
from xt_common import DATASETS, FIGDIR, SHORT, STEP1, STEP2  # noqa: E402

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.25})


def fig_only_groups():
    g = pd.read_parquet(STEP1 / 'cross_tokenizer_only_groups.parquet')
    t = g.groupby(['dataset', 'group']).size().unstack(fill_value=0)
    order = [c for c in ['Only-Spectral', 'Both', 'Only-RQVAE', 'Neither']
             if c in t.columns]
    t = t[order]
    frac = t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    bottom = np.zeros(len(frac))
    colors = {'Only-Spectral': '#1f77b4', 'Both': '#7f7f7f',
              'Only-RQVAE': '#d62728', 'Neither': '#dddddd'}
    for c in order:
        ax.bar([SHORT[d] for d in frac.index], frac[c], bottom=bottom,
               label=c, color=colors[c])
        for i, (b, v) in enumerate(zip(bottom, frac[c])):
            if v > 0.02:
                ax.text(i, b + v / 2, f'{100*v:.1f}%', ha='center', va='center',
                        fontsize=7, color='white' if c != 'Neither' else 'black')
        bottom += frac[c].to_numpy()
    ax.set_ylabel('fraction of test users (Hit@20)')
    ax.set_title('Cross-tokenizer Only groups: Spectral-K3 vs RQ-VAE-3layer')
    ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIGDIR / 'cross_tokenizer_only_groups.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote cross_tokenizer_only_groups.png')


def fig_fusion_oracle():
    d = pd.read_csv(STEP1 / 'cross_tokenizer_pair_diversity.csv')
    p = d[d.tier == 'PRIMARY']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(p))
    w = 0.38
    axes[0].bar(x - w / 2, p['best_single_ndcg20'], w, label='best single', color='#1f77b4')
    axes[0].bar(x + w / 2, p['fusion_ndcg20'], w, label='uniform fusion', color='#2ca02c')
    axes[0].plot(x, p['oracle_ndcg20'], 'k^', label='oracle', ms=6)
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT[d] for d in p['dataset']], rotation=15)
    axes[0].set_ylabel('NDCG@20'); axes[0].legend(fontsize=7); axes[0].set_title('retrieval')
    axes[1].bar(x - w / 2, p['fusion_gain'], w, color='#2ca02c', label='fusion gain')
    axes[1].bar(x + w / 2, p['oracle_gain'], w, color='#ff7f0e', label='oracle gain')
    axes[1].set_xticks(x); axes[1].set_xticklabels([SHORT[d] for d in p['dataset']], rotation=15)
    axes[1].set_ylabel('NDCG@20 gain over best single'); axes[1].legend(fontsize=7)
    axes[1].set_title('gains')
    axes[2].bar(x, p['hit_disagreement'], w, color='#d62728', label='Hit@20 disagreement')
    axes[2].bar(x, p['candidate_jaccard'], w, bottom=p['hit_disagreement'],
                color='#9467bd', label='candidate Jaccard')
    axes[2].set_xticks(x); axes[2].set_xticklabels([SHORT[d] for d in p['dataset']], rotation=15)
    axes[2].set_ylabel('fraction'); axes[2].legend(fontsize=7); axes[2].set_title('disagreement')
    fig.suptitle('Cross-tokenizer (independent models): Spectral-K3 vs RQ-VAE-3layer')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'cross_tokenizer_fusion_oracle.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote cross_tokenizer_fusion_oracle.png')


def fig_nll_alignment():
    p1 = STEP1 / 'cross_tokenizer_nll_alignment.csv'
    p2 = STEP2 / 'shared_nll_alignment.csv'
    frames = []
    if p1.exists():
        a = pd.read_csv(p1)
        a = a[(a.tier == 'PRIMARY') & (a.normalisation == 'raw')]
        a['source'] = 'independent'
        frames.append(a)
    if p2.exists():
        b = pd.read_csv(p2)
        b['source'] = 'shared'
        b['normalisation'] = 'raw'
        frames.append(b)
    if not frames:
        print('skip nll alignment figure (no data)')
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df['group'] != 'BOTH_Only_groups']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, src in zip(axes, ['independent', 'shared']):
        d = df[df['source'] == src]
        if not len(d):
            ax.set_visible(False)
            continue
        x = np.arange(len(d))
        ax.bar(x, d['argmin_match_accuracy'], color='#4c72b0')
        ax.axhline(0.5, color='crimson', ls='--', lw=1.1, label='random 50%')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SHORT[r.dataset]}\n{r.group}" for r in d.itertuples()],
                           rotation=45, fontsize=6.5, ha='right')
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('P(argmin NLL == only hitting tokenizer)')
        ax.set_title(f'{src} models')
        ax.legend(fontsize=7)
    fig.suptitle('Own-tokenizer target-NLL alignment on Only-tokenizer users')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'cross_tokenizer_nll_alignment.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote cross_tokenizer_nll_alignment.png')


def fig_independent_vs_shared():
    p = STEP2 / 'independent_vs_shared_factorization.csv'
    if not p.exists():
        print('skip independent-vs-shared figure (no data)')
        return
    d = pd.read_csv(p)
    d = d[d['metric'] != '_best_single_ndcg20']
    metrics = list(dict.fromkeys(d['metric_label']))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.6))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        dd = d[d['metric_label'] == m]
        x = np.arange(len(dd))
        ax.bar(x - 0.2, dd['independent_factors'], 0.4, label='independent', color='#1f77b4')
        ax.bar(x + 0.2, dd['shared_checkpoint'], 0.4, label='shared', color='#2ca02c')
        ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in dd['dataset']],
                                             rotation=20, fontsize=7)
        ax.set_title(m, fontsize=7.5)
        ax.legend(fontsize=6.5)
    fig.suptitle('Independent vs shared parameters, same factorization pair')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'independent_vs_shared.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote independent_vs_shared.png')


def fig_shared_single():
    p = STEP2 / 'shared_single_factorization_metrics.csv'
    if not p.exists():
        print('skip shared single factorization figure (no data)')
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    labs = [f"{SHORT[r.dataset]}\n{r.label}" for r in d.itertuples()]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d['independent_ndcg20'], 0.4, label='independent model', color='#1f77b4')
    ax.bar(x + 0.2, d['shared_ndcg20'], 0.4, label='shared checkpoint', color='#2ca02c')
    for i, r in enumerate(d.itertuples()):
        ax.text(i, max(r.independent_ndcg20, r.shared_ndcg20) * 1.02,
                f'{100*r.shared_over_independent_ndcg20:.0f}%', ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_ylabel('NDCG@20')
    ax.set_title('Single-factorization performance: independent vs shared parameters')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'shared_single_factorization.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote shared_single_factorization.png')


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_only_groups, fig_fusion_oracle, fig_nll_alignment,
               fig_independent_vs_shared, fig_shared_single):
        try:
            fn()
        except Exception as exc:
            print(f'{fn.__name__} failed: {exc}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
