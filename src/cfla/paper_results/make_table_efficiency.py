#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TABLE_EFFICIENCY -- measured quality vs measured inference cost (section 43).

Joins two independent artifacts:
  * quality   : outputs/final_hyperparam_efficiency/m5_rq5/TEST.csv  (scope=m5,
                strategies uniform / cfla)   -- TEST NDCG@20
  * cost      : the measured inference-complexity table produced by
                measure_inference_complexity.py -- ms/user, users/s, peak memory

Headline efficiency metric (section 30) is the RELATIVE NDCG@20 gain and the
LATENCY RATIO, never an absolute NDCG delta.

Nothing is recomputed except the two ratios the spec defines, both from the two
columns that sit next to them in the same row.
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
import hashlib
from pathlib import Path

REPO = Path(_REPO_ROOT + '')
F = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'
OUT = rp.OUTPUT_ROOT / 'paper_tables' / 'efficiency'
OUT.mkdir(parents=True, exist_ok=True)

COST_CANDIDATES = [
    F / 'complexity' / 'INFERENCE_COMPLEXITY_MEASURED.csv',
    F / 'complexity' / 'INFERENCE_COMPLEXITY.csv',
]
M5 = F / 'm5_rq5' / 'TEST.csv'

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
DISP = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
        'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}


def sha16(p: Path) -> str:
    if not p.is_file():
        return ''
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def pick_cost():
    for p in COST_CANDIDATES:
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        if any(r.get('status') == 'MEASURED' for r in rows):
            return p, rows
    return COST_CANDIDATES[0], []


def dataset_of(r):
    """The measured table has no `dataset` column; the cohort is named inside
    `input_artifacts` (…/data/<dataset>/<dataset>_t5_….npy).  Read it from there."""
    ia = r.get('input_artifacts') or ''
    for ds in DATASETS:
        if f'/data/{ds}/' in ia or f'{ds}_t5_' in ia:
            return ds
    return r.get('dataset') or ''


def cost_row(rows, ds, want):
    """Pick the row for one dataset whose method label matches `want`."""
    best = None
    for r in rows:
        if dataset_of(r) != ds or r.get('status') != 'MEASURED':
            continue
        lab = (r.get('method') or '').lower()
        if not all(w in lab for w in want):
            continue
        # prefer the native-length pool (that IS the M5 definition)
        score = 1 if 'native-length' in lab else 0
        if best is None or score > best[0]:
            best = (score, r)
    return best[1] if best else None


def quality(ds, strategy):
    if not M5.exists():
        return None
    for r in csv.DictReader(open(M5)):
        if (r['dataset'] == ds and r['split'] == 'test' and r.get('scope') == 'm5'
                and r['strategy'] == strategy and r['status'] == 'OK'
                and r.get('NDCG@20')):
            return float(r['NDCG@20'])
    return None


def main():
    # fail loudly (not with a raw traceback) if the frozen result inputs are absent
    for _p in [x for x in globals().get('INPUT_FILES', []) if isinstance(x, Path)]:
        if not _p.is_file():
            raise SystemExit(
                "FATAL: this builder consumes frozen result CSVs/artifacts produced by "
                "the paper pipeline; missing: " + str(_p) +
                "\nSet CFLA_OUTPUT_ROOT to the directory holding them.")

    cp, crow = pick_cost()
    rows, tex_rows, missing = [], [], []
    for ds in DATASETS:
        u = cost_row(crow, ds, ('uniform',))
        c = cost_row(crow, ds, ('cfla',))
        qu, qc = quality(ds, 'uniform'), quality(ds, 'cfla')
        rec = {'dataset': ds}
        rec['uniform_ndcg20'] = qu
        rec['cfla_ndcg20'] = qc
        rec['relative_gain_pct'] = ((qc - qu) / qu * 100.0
                                    if (qu and qc) else None)
        for tag, src in (('uniform', u), ('cfla', c)):
            if src is None:
                missing.append(f'{ds}/{tag}')
                rec[f'{tag}_ms_per_user'] = None
                rec[f'{tag}_users_per_sec'] = None
                rec[f'{tag}_peak_memory_MiB'] = None
                continue
            rec[f'{tag}_ms_per_user'] = fnum(src.get('total_ms_user'))
            rec[f'{tag}_users_per_sec'] = fnum(src.get('users_per_sec'))
            rec[f'{tag}_peak_memory_MiB'] = fnum(src.get('peak_memory_MiB'))
        mu, mc = rec['uniform_ms_per_user'], rec['cfla_ms_per_user']
        rec['latency_ratio'] = (mc / mu if (mu and mc) else None)
        rows.append(rec)

    fields = ['dataset', 'uniform_ndcg20', 'cfla_ndcg20', 'relative_gain_pct',
              'uniform_ms_per_user', 'cfla_ms_per_user', 'latency_ratio',
              'uniform_users_per_sec', 'cfla_users_per_sec',
              'uniform_peak_memory_MiB', 'cfla_peak_memory_MiB']
    with open(OUT / 'TABLE_EFFICIENCY.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r[k]) for k in fields})

    def g(r, k, fmt='{:.3f}'):
        v = r.get(k)
        return '--' if v is None else fmt.format(v)

    L = ['% AUTO-GENERATED by outputs/final_paper_completion/make_table_efficiency.py',
         '% No hand-entered numbers.',
         '% sources:',
         f'%   {M5.relative_to(REPO)}  sha256={sha16(M5)} (scope=m5 Uniform/CFLA, TEST)',
         f'%   {cp.relative_to(REPO)}  sha256={sha16(cp)} (measured latency)',
         '% Headline metrics: relative NDCG@20 gain and the CFLA/Uniform latency ratio.',
         r'\begin{table*}[t]', r'\centering', r'\small',
         r'\setlength{\tabcolsep}{4pt}',
         r'\caption{Quality versus measured inference cost. \textit{Rel.\ gain} is '
         r'$(\mathrm{CFLA}-\mathrm{Uniform})/\mathrm{Uniform}$ in TEST NDCG@20; '
         r'\textit{Lat.\ ratio} is CFLA latency over Uniform latency on the same '
         r'user cohort. Both methods score the same candidate union, so the extra '
         r'cost is the candidate-level teacher-forced likelihood pass plus its '
         r'calibration, not a larger candidate set. Lower latency ratio is better.}',
         r'\label{tab:efficiency}',
         r'\resizebox{\textwidth}{!}{%',
         r'\begin{tabular}{lcccccccc}',
         r'\toprule',
         r' & \multicolumn{2}{c}{TEST NDCG@20} & & \multicolumn{2}{c}{ms / user} & '
         r'& \multicolumn{2}{c}{users / s} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){8-9}',
         r'Dataset & Uniform & CFLA & Rel.\ gain (\%) & Uniform & CFLA & '
         r'Lat.\ ratio & Uniform & CFLA \\',
         r'\midrule']
    for r in rows:
        L.append(' & '.join([
            DISP[r['dataset']], g(r, 'uniform_ndcg20', '{:.4f}'),
            g(r, 'cfla_ndcg20', '{:.4f}'), g(r, 'relative_gain_pct', '{:+.2f}'),
            g(r, 'uniform_ms_per_user'), g(r, 'cfla_ms_per_user'),
            g(r, 'latency_ratio', '{:.3f}x'),
            g(r, 'uniform_users_per_sec', '{:.1f}'),
            g(r, 'cfla_users_per_sec', '{:.1f}')]) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    (OUT / 'TABLE_EFFICIENCY.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    print(f'wrote {OUT / "TABLE_EFFICIENCY.csv"} and .tex')
    print(f'  cost source: {cp.name}  ({len(crow)} rows, '
          f'{sum(1 for r in crow if r.get("status") == "MEASURED")} measured)')
    for r in rows:
        print(f"  {DISP[r['dataset']]:12s} ndcg {r['uniform_ndcg20']} -> "
              f"{r['cfla_ndcg20']}  gain {r['relative_gain_pct']}  "
              f"lat ratio {r['latency_ratio']}")
    if missing:
        print(f'  MISSING cost rows for: {missing}')
    return 0 if not missing else 2


if __name__ == '__main__':
    raise SystemExit(main())
