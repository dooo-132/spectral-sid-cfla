#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIG_CODEBOOK_SENSITIVITY -- Beauty Spectral S4, CFLA vs Uniform over codebook size.

x: C in {32, 64, 128, 256, 512} on a log2 axis.
y: relative NDCG@20 improvement of CFLA over Uniform, in percent.
A horizontal zero line is drawn.  Points whose paired-bootstrap 95% interval for
the ABSOLUTE NDCG difference excludes zero are filled; the rest are drawn hollow
and labelled non-significant -- none is dropped.

Scale note: the frozen bootstrap resampled the ABSOLUTE NDCG difference, so the
interval is NOT rescaled onto the relative axis here (the denominator varies
across resamples, so a rescaled interval would be a statistic that was never
computed).  Significance is sign-invariant, so it transfers exactly.
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
import csv
import sys
from pathlib import Path

REPO = Path(_REPO_ROOT + '')
STYLE = Path(__file__).resolve().parent
OUT = rp.OUTPUT_ROOT / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(STYLE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fs.OUT = OUT
fs.apply_style()

CAND = [
    rp.OUTPUT_ROOT / 'final_paper_completion' / 'S4_CODEBOOK_CFLA_FINAL.csv',
    rp.OUTPUT_ROOT / 'final_hyperparam_efficiency' / 'codebook' / 'S4_CODEBOOK_CFLA_FINAL.csv',
    rp.OUTPUT_ROOT / 'final_hyperparam_efficiency' / 'codebook' / 'CFLA_CODEBOOK_RESULTS.csv',
]
BOOT = [
    rp.OUTPUT_ROOT / 'final_paper_completion' / 'S4_CODEBOOK_CFLA_BOOTSTRAP.csv',
    rp.OUTPUT_ROOT / 'final_hyperparam_efficiency' / 'codebook' / 'S4_CODEBOOK_CFLA_BOOTSTRAP.csv',
    rp.OUTPUT_ROOT / 'final_hyperparam_efficiency' / 'codebook' / 'CFLA_BOOTSTRAP.csv',
]


def first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def col(row, *names):
    for n in names:
        for k in row:
            if k.lower().replace('@', '').replace('_', '') == n.lower().replace('@', '').replace('_', ''):
                return row[k]
    return None


def main():
    # fail loudly (not with a raw traceback) if the frozen result inputs are absent
    for _p in [x for x in globals().get('INPUT_FILES', []) if isinstance(x, Path)]:
        if not _p.is_file():
            raise SystemExit(
                "FATAL: this builder consumes frozen result CSVs/artifacts produced by "
                "the paper pipeline; missing: " + str(_p) +
                "\nSet CFLA_OUTPUT_ROOT to the directory holding them.")

    fp, bp = first(CAND), first(BOOT)
    if fp is None:
        print('NO CODEBOOK RESULT FILE FOUND; looked at:')
        for p in CAND:
            print('  ', p)
        return 1
    rows = list(csv.DictReader(open(fp)))
    print(f'source: {fp}  ({len(rows)} rows)')
    print('columns:', list(rows[0].keys()))

    # keep TEST, prefer a row that already carries a relative improvement
    # the artifact is LONG format: one row per (codebook, method, split)
    per = {}
    for r in rows:
        if (col(r, 'split') or 'test').lower() != 'test':
            continue
        try:
            c = int(float(col(r, 'codebook', 'C', 'codebook_size')))
        except (TypeError, ValueError):
            continue
        m = (col(r, 'method') or '').lower()
        v = col(r, 'NDCG@20', 'ndcg20')
        if v in (None, ''):
            continue
        per.setdefault(c, {})[m] = float(v)
    data = {}
    for c, d in per.items():
        uni = next((v for k, v in d.items() if 'uniform' in k), None)
        cfla = next((v for k, v in d.items()
                     if 'cfla' in k or 'lse' in k), None)
        if uni is None or cfla is None:
            continue
        data[c] = {'gain': (cfla - uni) / uni * 100.0, 'uni': uni, 'cfla': cfla}
    if not data:
        print('no TEST rows with a relative improvement could be parsed')
        return 2

    sig = {}
    if bp is not None:
        for r in csv.DictReader(open(bp)):
            if (col(r, 'metric') or '').lower() not in ('ndcg@20', 'ndcg20'):
                continue
            if (col(r, 'split') or 'test').lower() != 'test':
                continue
            c = col(r, 'codebook', 'C', 'codebook_size')
            if c is None:
                continue
            lo, hi = col(r, 'ci_lo'), col(r, 'ci_hi')
            if lo in (None, '', 'NOT_MEASURED') or hi in (None, '', 'NOT_MEASURED'):
                continue
            sig[int(float(c))] = not (float(lo) <= 0.0 <= float(hi))
        print(f'bootstrap: {bp}  significance resolved for C = {sorted(sig)}')

    cs = sorted(data)
    gains = [data[c]['gain'] for c in cs]

    fig, ax = plt.subplots(figsize=fs.FIGSIZE_1COL)
    x = np.arange(len(cs))
    for i, c in enumerate(cs):
        s = sig.get(c)
        filled = bool(s) if c in sig else True
        ax.bar(i, gains[i], 0.62,
               color=fs.OKABE_ITO['blue'] if filled else 'white',
               edgecolor=fs.OKABE_ITO['blue'], linewidth=1.1)
        tag = f'{gains[i]:+.2f}'
        if c in sig and not sig[c]:
            tag += '\n(n.s.)'
        ax.annotate(tag, (i, gains[i]), textcoords='offset points',
                    xytext=(0, 2.0 if gains[i] >= 0 else -9.0), ha='center',
                    fontsize=6.3)
    ax.axhline(0.0, color='#333333', linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel('Codebook size $C$')
    ax.set_ylabel('Relative NDCG@20 gain of CFLA (%)')
    lo = min(0.0, min(gains))
    hi = max(0.0, max(gains))
    pad = max(0.15 * (hi - lo), 0.5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    paths = fs.save(fig, 'FIG_CODEBOOK_SENSITIVITY')

    src = []
    for c in cs:
        d = data[c]
        src.append(dict(codebook=c, split='test',
                        uniform_ndcg20='' if d['uni'] is None else repr(d['uni']),
                        cfla_ndcg20='' if d['cfla'] is None else repr(d['cfla']),
                        relative_gain_pct=repr(d['gain']),
                        significant_ci_excludes_zero=sig.get(c, 'NOT_AVAILABLE')))
    fs.write_source_csv(src, OUT / 'FIG_CODEBOOK_SENSITIVITY_SOURCE.csv')
    fs.audit_rows([dict(artifact='FIG_CODEBOOK_SENSITIVITY',
                        source_file=str(fp), source_status='AUTHORITATIVE',
                        dataset='Beauty', metric='NDCG@20',
                        method=f'CFLA minus Uniform at C={s["codebook"]}',
                        value=s['relative_gain_pct'], used_in='bar')
                   for s in src],
                  OUT / 'FIG_CODEBOOK_SENSITIVITY_SOURCE_AUDIT.csv')

    print('wrote:')
    for p in paths:
        print('  ', p, p.stat().st_size, 'B')
    print('  ', OUT / 'FIG_CODEBOOK_SENSITIVITY_SOURCE.csv')
    for c in cs:
        print(f"   C={c:<4d} gain={data[c]['gain']:+.3f}%  "
              f"sig={sig.get(c, 'n/a')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
