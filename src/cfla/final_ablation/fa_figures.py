#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Required figures for the final ablation suite (section 63).

These are ANALYSIS figures, deliberately not named "paper_figure_*": none of them is a
publication figure and they must not be lifted into the paper without review.

    python3 fa_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fa_common as C  # noqa: E402

FIG = C.OUT / 'figures'
DS_ORDER = C.DATASETS + [C.SPORTS_DS]
LABEL = {**C.SHORT, C.SPORTS_DS: 'Sports*'}
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']


def rd(name):
    p = C.OUT / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def _bar_axes(ax, datasets):
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([LABEL[d] for d in datasets], rotation=20, ha='right')


def fig_calibration():
    rows = rd('calibration_ablation.csv')
    if not rows:
        print('  skip calibration (no data)'); return
    methods = ['A0_Uniform', 'A1_RawLSE', 'A2_MeanLSE', 'A4_CenteredLSE', 'A3_ZLSE']
    nice = ['Uniform', 'Raw LSE', 'Mean-token LSE', 'Mean-centred LSE', 'z-LSE (main)']
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.16
    for i, (m, lab) in enumerate(zip(methods, nice)):
        vals = []
        for ds in DS_ORDER:
            r = [x for x in rows if x['dataset'] == ds and x['method'] == m]
            vals.append(float(r[0]['NDCG@20']) if r else np.nan)
        ax.bar(np.arange(len(DS_ORDER)) + (i - 2) * w, vals, w, label=lab,
               color=COLORS[i], edgecolor='black', linewidth=0.4)
    _bar_axes(ax, DS_ORDER)
    ax.set_ylabel('NDCG@20')
    ax.set_title('Ablation A — NLL calibration\n(* Sports is the confirmatory dataset)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / 'analysis_calibration_ablation.png', dpi=150)
    plt.close(fig); print('  analysis_calibration_ablation.png')


def fig_aggregation():
    rows = rd('aggregation_ablation.csv')
    if not rows:
        print('  skip aggregation (no data)'); return
    methods = ['B1_MinNLL', 'B2_AvgNLL', 'B3_ZLSE', 'B4_WorstNLL']
    nice = ['Min (best factorization)', 'Average', 'LSE (main)', 'Max (worst) [control]']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.2
    for i, (m, lab) in enumerate(zip(methods, nice)):
        vals = [float([x for x in rows if x['dataset'] == ds and x['method'] == m][0]['NDCG@20'])
                for ds in DS_ORDER]
        axes[0].bar(np.arange(len(DS_ORDER)) + (i - 1.5) * w, vals, w, label=lab,
                    color=COLORS[i], edgecolor='black', linewidth=0.4)
    _bar_axes(axes[0], DS_ORDER)
    axes[0].set_ylabel('NDCG@20'); axes[0].set_title('Absolute NDCG@20')
    axes[0].legend(fontsize=8)

    # deltas vs LSE make the small differences legible
    for i, (m, lab) in enumerate(zip(methods, nice)):
        d = []
        for ds in DS_ORDER:
            z = float([x for x in rows if x['dataset'] == ds and x['method'] == 'B3_ZLSE'][0]['NDCG@20'])
            v = float([x for x in rows if x['dataset'] == ds and x['method'] == m][0]['NDCG@20'])
            d.append(v - z)
        axes[1].bar(np.arange(len(DS_ORDER)) + (i - 1.5) * w, d, w, label=lab,
                    color=COLORS[i], edgecolor='black', linewidth=0.4)
    axes[1].axhline(0, color='black', linewidth=0.8)
    _bar_axes(axes[1], DS_ORDER)
    axes[1].set_ylabel('NDCG@20 minus LSE')
    axes[1].set_title('Difference from LSE')
    fig.suptitle('Ablation B — aggregation function', y=1.00)
    fig.tight_layout(); fig.savefig(FIG / 'analysis_aggregation_ablation.png', dpi=150)
    plt.close(fig); print('  analysis_aggregation_ablation.png')


def fig_shared_vs_independent():
    rows = rd('shared_vs_independent.csv')
    cal = rd('calibration_ablation.csv')
    if not rows:
        print('  skip shared-vs-independent (no data yet)'); return
    ind = {}
    for ds in C.DATASETS:
        u = [x for x in rows if x['dataset'] == ds and x['method'].startswith('Uniform')]
        l = [x for x in rows if x['dataset'] == ds and 'LSE' in x['method']]
        if u and l:
            ind[ds] = float(l[0]['NDCG@20']) - float(u[0]['NDCG@20'])
    shared = {}
    for ds in C.DATASETS:
        u = [x for x in cal if x['dataset'] == ds and x['method'] == 'A0_Uniform']
        z = [x for x in cal if x['dataset'] == ds and x['method'] == 'A3_ZLSE']
        if u and z:
            shared[ds] = float(z[0]['NDCG@20']) - float(u[0]['NDCG@20'])
    ds_list = [d for d in C.DATASETS if d in ind and d in shared]
    if not ds_list:
        print('  skip shared-vs-independent (no overlapping datasets)'); return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(ds_list)); w = 0.36
    ax.bar(x - w / 2, [shared[d] for d in ds_list], w, label='Shared model',
           color='#4C72B0', edgecolor='black', linewidth=0.4)
    ax.bar(x + w / 2, [ind[d] for d in ds_list], w, label='Independent models',
           color='#DD8452', edgecolor='black', linewidth=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in ds_list], rotation=20, ha='right')
    ax.set_ylabel('LSE − Uniform (NDCG@20)')
    ax.set_title('Ablation D — within-regime gain, shared vs independent\n'
                 'raw NDCG is not comparable across regimes; only the gain is')
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / 'analysis_shared_vs_independent_gain.png', dpi=150)
    plt.close(fig); print('  analysis_shared_vs_independent_gain.png')


def fig_beam_budget():
    rows = rd('beam_budget_ablation.csv')
    cnt = rd('beam_budget_count_matched.csv')
    if not rows:
        print('  skip beam budget (no data)'); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, ds in enumerate(DS_ORDER):
        rs = sorted([x for x in rows if x['dataset'] == ds], key=lambda r: int(r['beam_B']))
        B = [int(r['beam_B']) for r in rs]
        axes[0].plot(B, [float(r['Uniform_NDCG@20']) for r in rs], 'o-', color='#4C72B0', label='Uniform' if i == 0 else None)
        axes[0].plot(B, [float(r['ZLSE_NDCG@20']) for r in rs], 's-', color='#C44E52', label='z-LSE' if i == 0 else None)
        axes[1].plot(B, [float(r['ZLSE_NDCG@20']) - float(r['Uniform_NDCG@20']) for r in rs], 'o-', label=LABEL[ds])
        axes[2].plot(B, [float(r['coverage']) for r in rs], 'o-', label=LABEL[ds])
    axes[0].set_xlabel('beam budget B'); axes[0].set_ylabel('NDCG@20')
    axes[0].set_title('Absolute'); axes[0].legend()
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('beam budget B'); axes[1].set_ylabel('LSE − Uniform')
    axes[1].set_title('Within-regime gain'); axes[1].legend(fontsize=8)
    axes[2].set_xlabel('beam budget B'); axes[2].set_ylabel('target union coverage')
    axes[2].set_title('Coverage'); axes[2].legend(fontsize=8)
    fig.suptitle('Ablation F — beam / candidate budget', y=1.02)
    fig.tight_layout(); fig.savefig(FIG / 'analysis_beam_budget_curve.png', dpi=150)
    plt.close(fig); print('  analysis_beam_budget_curve.png')


def fig_subgroup():
    rows = rd('subgroup_analysis.csv')
    if not rows:
        print('  skip subgroup (no data)'); return
    groups = ['Only_F1', 'Only_F2', 'Both', 'Neither']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.2
    for i, g in enumerate(groups):
        d = []
        for ds in DS_ORDER:
            u = [x for x in rows if x['dataset'] == ds and x['group'] == g and x['method'] == 'Uniform']
            z = [x for x in rows if x['dataset'] == ds and x['group'] == g and x['method'] == 'ZLSE']
            d.append(float(z[0]['NDCG@20']) - float(u[0]['NDCG@20']) if (u and z) else np.nan)
        axes[0].bar(np.arange(len(DS_ORDER)) + (i - 1.5) * w, d, w, label=g, color=COLORS[i],
                    edgecolor='black', linewidth=0.4)
    axes[0].axhline(0, color='black', linewidth=0.8)
    _bar_axes(axes[0], DS_ORDER)
    axes[0].set_ylabel('NDCG@20 change (LSE − Uniform)')
    axes[0].set_title('Gain by subgroup'); axes[0].legend(fontsize=8)

    for i, ds in enumerate(DS_ORDER):
        vals = []
        for g in groups:
            r = [x for x in rows if x['dataset'] == ds and x['group'] == g and x['method'] == 'ZLSE']
            vals.append(float(r[0]['n']) if r else 0.0)
        tot = sum(vals) or 1.0
        axes[1].bar(i, vals[0] / tot, color=COLORS[0], edgecolor='black', linewidth=0.4,
                    label='Only_F1' if i == 0 else None)
        bot = vals[0] / tot
        for j in (1, 2, 3):
            axes[1].bar(i, vals[j] / tot, bottom=bot, color=COLORS[j],
                        edgecolor='black', linewidth=0.4,
                        label=groups[j] if i == 0 else None)
            bot += vals[j] / tot
    axes[1].set_xticks(range(len(DS_ORDER)))
    axes[1].set_xticklabels([LABEL[d] for d in DS_ORDER], rotation=20, ha='right')
    axes[1].set_ylabel('fraction of users')
    axes[1].set_title('Subgroup composition'); axes[1].legend(fontsize=8)
    fig.suptitle('Ablation H — where the gain comes from', y=1.02)
    fig.tight_layout(); fig.savefig(FIG / 'analysis_subgroup_gain.png', dpi=150)
    plt.close(fig); print('  analysis_subgroup_gain.png')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print(f'writing figures to {FIG}')
    fig_calibration()
    fig_aggregation()
    fig_shared_vs_independent()
    fig_beam_budget()
    fig_subgroup()
    return 0


if __name__ == '__main__':
    sys.exit(main())
