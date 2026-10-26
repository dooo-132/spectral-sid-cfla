#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TABLE 1 (final) -- overall recommendation performance, length-matched singles.

Method set (section 32)
  Dimension Split      Even        = Even Split K5        (L = 6)
                       Random      = Random Split K5      (L = 6)
  Quantization-based   RQ-VAE      = RQ-VAE5              (L = 6, C = 256)
                       RQ-OPQ      = RQ2-OPQ3            (L = 6, C = 256)
                                         2 RQ semantic positions + 3 OPQ semantic
                                         positions + 1 disambiguation token.
                                         OPQ5 is NOT used: 768 % 5 != 0
  Spectral-based       Spectral    = Spectral-K5          (L = 6)
  Cross-Factorization  CF-Uniform  = M5 Uniform           (5 arms, native lengths)
                       CF-LA       = M5 CFLA              (same pool, same union)

EVERY SINGLE-MODEL BASELINE IS LENGTH-MATCHED at 5 semantic + 1 disambiguation
token.  CF-Uniform and CF-LA are NOT length-matched: they fuse a heterogeneous
native-length pool (Spectral K2/K3/K4/K5 at L = 3/4/5/6 plus RQ-VAE5 at L = 6).

Marking (sections 34-35)
  dagger    on the best SINGLE baseline of each dataset x metric, computed per
            metric -- never one fixed single winner per dataset
  bold      overall best across all seven methods
  underline overall second
  CF-LA is NOT forced to be bold.

Improvement columns, in the paper's order (section 36)
  Imp. vs Best    = (CF-LA - best_non_cfla) / best_non_cfla * 100
                    best_non_cfla = max(five singles, CF-Uniform)
  Imp. vs Single  = (CF-LA - best_single)   / best_single   * 100
                    best_single   = max(five singles)

Every number is read from a frozen artifact; nothing is hand-entered.
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
import json
from pathlib import Path

REPO = Path(_REPO_ROOT + '')
F = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'
OUT = rp.OUTPUT_ROOT / 'paper_tables' / 'table1_final'
OUT.mkdir(parents=True, exist_ok=True)

SINGLES_A = F / 'single_5plus1' / 'TEST_RESULTS.csv'
SINGLES_B = F / 'single_5plus1' / 'TEST_RESULTS_FINAL.csv'
M5 = F / 'm5_rq5' / 'TEST.csv'

DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
DISPLAY_DS = {'Beauty': 'Beauty', 'Cell_Phones_and_Accessories': 'Cell Phones',
              'Health_and_Personal_Care': 'Health', 'Toys_and_Games': 'Toys'}

SHOWN = ['Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10']
EXTRA = ['Recall@20', 'NDCG@20']
METRIC_TEX = {'Recall@5': 'R@5', 'Recall@10': 'R@10', 'Recall@20': 'R@20',
              'NDCG@5': 'N@5', 'NDCG@10': 'N@10', 'NDCG@20': 'N@20'}

COLS = ['even', 'random', 'rqvae', 'rqopq', 'spectral', 'cf_uniform', 'cf_la']
COL_TEX = {'even': 'Even', 'random': 'Random', 'rqvae': 'RQ-VAE', 'rqopq': 'RQ-OPQ',
           'spectral': 'Spectral', 'cf_uniform': 'CF-Uni.', 'cf_la': 'CF-LA'}
SINGLE_COLS = ['even', 'random', 'rqvae', 'rqopq', 'spectral']
FULL_NAME = {'even': 'Even Split K5 (L=6)', 'random': 'Random Split K5 (L=6)',
             'rqvae': 'RQ-VAE5 (L=6, C=256)',
             'rqopq': 'RQ2-OPQ3 (L=6, C=256; 2 RQ + 3 OPQ + 1 disamb.)',
             'spectral': 'Spectral-K5 (L=6, nonneg_pearson_v3)',
             'cf_uniform': 'M5 Uniform (5-arm native-length pool)',
             'cf_la': 'M5 CFLA (same 5-arm pool and union)'}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_singles():
    """{dataset: {col: {metric: value}}} + the raw source row."""
    out = {d: {} for d in DATASETS}
    raw = {}
    for r in csv.DictReader(open(SINGLES_A)):
        if r['dataset'] not in DATASETS or r['split'] != 'test':
            continue
        m = r['method']
        if m == 'Even Split K5':
            col = 'even'
        elif m == 'Random Split K5':
            col = 'random'
        elif m == 'Spectral K5':
            if r.get('affinity') != 'nonneg_pearson_v3':
                continue
            col = 'spectral'
        else:
            continue
        # a single-model row counts only if it actually carries metrics; the
        # un-generated arms (OPQ5, RQ1-OPQ4) are status NOT_GENERATED with blanks
        if not r.get('NDCG@20'):
            continue
        vals = {k: float(r[k]) for k in SHOWN + EXTRA if r.get(k)}
        if len(vals) != len(SHOWN + EXTRA):
            raise SystemExit(f'{r["dataset"]}/{m}: incomplete metrics {sorted(vals)}')
        out[r['dataset']][col] = vals
        raw[(r['dataset'], col)] = ('TEST_RESULTS.csv', m)
    for r in csv.DictReader(open(SINGLES_B)):
        if r['dataset'] not in DATASETS or r['split'] != 'test':
            continue
        col = {'rqvae5': 'rqvae', 'rq2_opq3': 'rqopq'}.get(r['method'])
        if col is None:
            continue
        vals = {k: float(r[k]) for k in SHOWN + EXTRA if r.get(k)}
        if len(vals) != len(SHOWN + EXTRA):
            raise SystemExit(f'{r["dataset"]}/{r["method"]}: incomplete metrics')
        out[r['dataset']][col] = vals
        raw[(r['dataset'], col)] = ('TEST_RESULTS_FINAL.csv', r['method'])
    for d in DATASETS:
        missing = [c for c in SINGLE_COLS if c not in out[d]]
        if missing:
            raise SystemExit(f'{d}: missing single baselines {missing}')
    return out, raw


def load_m5():
    """{dataset: {col: {metric: value}}} for the scope == 'm5' rows."""
    out = {d: {} for d in DATASETS}
    for r in csv.DictReader(open(M5)):
        if r['dataset'] not in DATASETS or r['split'] != 'test':
            continue
        if r.get('scope') != 'm5':
            continue
        col = {'uniform': 'cf_uniform', 'cfla': 'cf_la'}.get(r['strategy'])
        if col is None:
            continue
        if r['status'] != 'OK':
            raise SystemExit(
                f'{r["dataset"]}/{r["strategy"]}: status is {r["status"]!r}, not OK. '
                f'The final Table 1 must not be built while CFLA is unmeasured.')
        vals = {k: float(r[k]) for k in SHOWN + EXTRA if r.get(k)}
        if len(vals) != len(SHOWN + EXTRA):
            raise SystemExit(f'{r["dataset"]}/{r["strategy"]}: incomplete metrics '
                             f'{sorted(vals)}')
        if col in out[r['dataset']]:
            raise SystemExit(f'{r["dataset"]}/{col}: two scope=m5 rows -- ambiguous')
        out[r['dataset']][col] = vals
    for d in DATASETS:
        for c in ('cf_uniform', 'cf_la'):
            if c not in out[d]:
                raise SystemExit(f'{d}: missing {c} on the m5 scope')
    return out


def tex_num(v: float, bold=False, under=False, dag=False) -> str:
    s = f'{v:.4f}'
    if bold:
        s = r'\textbf{' + s + '}'
    elif under:
        s = r'\underline{' + s + '}'
    if dag:
        s = s + r'$^{\dagger}$'
    return s


def _require_inputs():
    missing = [str(p) for p in (SINGLES_A, SINGLES_B, M5) if not p.is_file()]
    if missing:
        raise SystemExit(
            "FATAL: this table builder consumes the frozen result CSVs produced by "
            "the paper pipeline; missing: " + ", ".join(missing) +
            "\nSet CFLA_OUTPUT_ROOT to the directory holding "
            "final_hyperparam_efficiency/ or run the pipeline first "
            "(docs/PIPELINE_STAGES.md, stage 6).")


def main():
    _require_inputs()
    singles, raw = load_singles()
    m5 = load_m5()

    full_rows, disp_rows, audit, bsaudit = [], [], [], []
    for ds in DATASETS:
        for metric in SHOWN + EXTRA:
            vals = {c: singles[ds][c][metric] for c in SINGLE_COLS}
            vals['cf_uniform'] = m5[ds]['cf_uniform'][metric]
            vals['cf_la'] = m5[ds]['cf_la'][metric]

            best_single_col = max(SINGLE_COLS, key=lambda c: vals[c])
            best_single = vals[best_single_col]
            non_cfla = {c: vals[c] for c in COLS if c != 'cf_la'}
            best_non_cfla_col = max(non_cfla, key=lambda c: non_cfla[c])
            best_non_cfla = non_cfla[best_non_cfla_col]

            order = sorted(COLS, key=lambda c: -vals[c])
            overall_best, overall_second = order[0], order[1]

            imp_best = (vals['cf_la'] - best_non_cfla) / best_non_cfla * 100.0
            imp_single = (vals['cf_la'] - best_single) / best_single * 100.0

            full_rows.append(dict(
                dataset=ds, metric=metric,
                **{c: repr(vals[c]) for c in COLS},
                best_single=repr(best_single), best_single_method=best_single_col,
                best_non_cfla=repr(best_non_cfla),
                best_non_cfla_method=best_non_cfla_col,
                improv_vs_best_pct=repr(imp_best),
                improv_vs_single_pct=repr(imp_single),
                overall_best=overall_best, overall_second=overall_second))
            disp_rows.append(dict(
                dataset=DISPLAY_DS[ds], metric=METRIC_TEX[metric],
                **{COL_TEX[c]: f'{vals[c]:.4f}' for c in COLS},
                best_single_method=COL_TEX[best_single_col],
                improv_vs_best=f'{imp_best:+.2f}%',
                improv_vs_single=f'{imp_single:+.2f}%',
                overall_best=COL_TEX[overall_best],
                overall_second=COL_TEX[overall_second]))

            for c in COLS:
                src_file, src_method = (
                    ('m5_rq5/TEST.csv', 'uniform' if c == 'cf_uniform' else 'cfla')
                    if c.startswith('cf_') else raw[(ds, c)])
                audit.append(dict(dataset=ds, metric=metric, column=c,
                                  method=FULL_NAME[c], value=repr(vals[c]),
                                  source_file=src_file, source_method=src_method,
                                  source_status='AUTHORITATIVE_FROZEN'))
            bsaudit.append(dict(
                dataset=ds, metric=metric,
                candidates='|'.join(f'{c}={vals[c]:.6f}' for c in SINGLE_COLS),
                best_single_method=best_single_col,
                best_single_value=repr(best_single),
                best_non_cfla_method=best_non_cfla_col,
                best_non_cfla_value=repr(best_non_cfla),
                rule='per dataset x metric maximum; NOT one fixed winner per dataset'))

    # ------------------------------------------------------------- CSVs ----
    with open(OUT / 'TABLE1_FINAL_FULL_PRECISION.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)
    with open(OUT / 'TABLE1_FINAL_DISPLAY.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(disp_rows[0].keys()))
        w.writeheader()
        w.writerows(disp_rows)
    with open(OUT / 'TABLE1_SOURCE_AUDIT.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0].keys()))
        w.writeheader()
        w.writerows(audit)
    with open(OUT / 'TABLE1_BEST_SINGLE_AUDIT.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(bsaudit[0].keys()))
        w.writeheader()
        w.writerows(bsaudit)

    # ------------------------------------------------------------- LaTeX ---
    by_ds = {}
    for r in full_rows:
        by_ds.setdefault(r['dataset'], {})[r['metric']] = r

    L = []
    L.append('% AUTO-GENERATED by outputs/final_paper_completion/build_table1_final.py')
    L.append('% No hand-entered numbers.  Required packages: booktabs, multirow, graphicx.')
    L.append('%')
    L.append('% Every SINGLE-model baseline is length-matched at 5 semantic + 1')
    L.append('% disambiguation token (L = 6).  CF-Uniform / CF-LA are NOT length-matched:')
    L.append('% they fuse the heterogeneous native-length pool')
    L.append('% Spectral K2/K3/K4/K5 (L = 3/4/5/6) + RQ-VAE5 (L = 6).')
    L.append('% RQ-OPQ is RQ2-OPQ3 (2 RQ + 3 OPQ semantic positions + 1 disambiguation).')
    L.append('% OPQ5 is not used: 768 %% 5 != 0.')
    L.append('%')
    L.append(f'% sources: {SINGLES_A.relative_to(REPO)}  sha256={sha(SINGLES_A)[:16]}')
    L.append(f'%          {SINGLES_B.relative_to(REPO)}  sha256={sha(SINGLES_B)[:16]}')
    L.append(f'%          {M5.relative_to(REPO)}  sha256={sha(M5)[:16]} (scope=m5)')
    L.append(r'\begin{table*}[t]')
    L.append(r'\centering')
    L.append(r'\setlength{\tabcolsep}{4.2pt}')
    L.append(r'\renewcommand{\arraystretch}{1.06}')
    L.append(r'\caption{Overall test-set recommendation performance on the four '
             r'development datasets. Every \emph{single}-model baseline is '
             r'length-matched at five semantic tokens plus one disambiguation token '
             r'($L{=}6$): Even Split $K{=}5$, Random Split $K{=}5$, RQ-VAE5, '
             r'RQ2-OPQ3 and Spectral-$K{=}5$. \textit{RQ-OPQ} denotes RQ2-OPQ3 (two '
             r'residual-quantisation semantic positions, three OPQ semantic '
             r'positions and one disambiguation token); OPQ5 is not a legal OPQ '
             r'construction at this embedding dimension ($768 \bmod 5 \neq 0$) and '
             r'is not used. \textit{CF-Uni.} (Uniform probability fusion) and '
             r'\textit{CF-LA} (Cross-Factorization Likelihood Aggregation) are '
             r'\emph{not} length-matched: they fuse the same heterogeneous '
             r'native-length pool, Spectral $K\in\{2,3,4,5\}$ ($L{=}3,4,5,6$) plus '
             r'RQ-VAE5 ($L{=}6$), over the identical candidate union. '
             r'$\dagger$ marks the best single-model baseline of that row, chosen '
             r'per metric. Best overall in bold, second best underlined.}')
    L.append(r'\label{tab:overall}')
    L.append(r'\resizebox{\textwidth}{!}{')
    L.append(r'\begin{tabular}{llcccccccrr}')   # 2 + 7 methods + 2 improv = 11
    L.append(r'\toprule')
    L.append(r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Metric} & '
             r'\multicolumn{2}{c}{\textit{Dimension Split}} & '
             r'\multicolumn{2}{c}{\textit{Quantization}} & '
             r'\multicolumn{1}{c}{\textit{Spectral}} & '
             r'\multicolumn{2}{c}{\textit{Cross-Factorization}} & '
             r'\multicolumn{2}{c}{Improv.} \\')
    L.append(r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}'
             r'\cmidrule(lr){8-9}\cmidrule(lr){10-11}')
    L.append(r' & & Even & Random & RQ-VAE & RQ-OPQ & Spectral & CF-Uni. & CF-LA & '
             r'vs.\ Best & vs.\ Single \\')
    L.append(r'\midrule')
    for di, ds in enumerate(DATASETS):
        if di:
            L.append(r'\midrule')
        n = len(SHOWN)
        for mi, metric in enumerate(SHOWN):
            r = by_ds[ds][metric]
            cells = []
            for c in COLS:
                v = float(r[c])
                cells.append(tex_num(v,
                                     bold=(r['overall_best'] == c),
                                     under=(r['overall_second'] == c),
                                     dag=(c in SINGLE_COLS
                                          and c == r['best_single_method'])))
            first = (r'\multirow{%d}{*}{%s} & %s' % (n, DISPLAY_DS[ds],
                                                      METRIC_TEX[metric])
                     if mi == 0 else r' & %s' % METRIC_TEX[metric])
            L.append(f'{first} & ' + ' & '.join(cells)
                     + f" & {float(r['improv_vs_best_pct']):+.2f}\\%"
                     + f" & {float(r['improv_vs_single_pct']):+.2f}\\% \\\\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'}')
    L.append(r'\end{table*}')
    (OUT / 'TABLE1_FINAL.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    # ------------------------------------------------------------- report --
    rep = []
    rep.append('# TABLE1_FINAL_REPORT.md\n')
    rep.append('```\nSTATUS = COMPLETE\n```\n')
    rep.append('Final Table 1 for the paper. No hand-entered numbers; every cell is '
               'read from a frozen artifact.\n')
    rep.append('## Method set and length matching\n')
    rep.append('| column | method | length-matched? |')
    rep.append('|---|---|---|')
    for c in COLS:
        lm = 'YES (5 semantic + 1 disambiguation, L=6)' if c in SINGLE_COLS \
            else 'NO (native-length heterogeneous pool)'
        rep.append(f'| {COL_TEX[c]} | {FULL_NAME[c]} | {lm} |')
    rep.append('')
    rep.append('RQ-OPQ is **RQ2-OPQ3**: 2 RQ semantic positions + 3 OPQ semantic '
               'positions + 1 disambiguation token. OPQ5 is not used '
               '(768 mod 5 != 0).\n')
    rep.append('## Marking rules applied\n')
    rep.append('* dagger: best single baseline **per dataset x metric** '
               '(not one fixed winner per dataset)\n'
               '* bold: best overall across all seven methods\n'
               '* underline: second best overall\n'
               '* CF-LA is not forced to be bold\n')
    rep.append('## Improvement columns (paper order)\n')
    rep.append('* `Imp. vs Best` = (CF-LA - max(five singles, CF-Uni.)) / that * 100\n'
               '* `Imp. vs Single` = (CF-LA - max(five singles)) / that * 100\n')
    rep.append('## How often is CF-LA the overall best?\n')
    n_la = sum(1 for r in full_rows if r['overall_best'] == 'cf_la')
    n_uni = sum(1 for r in full_rows if r['overall_best'] == 'cf_uniform')
    rep.append(f'* CF-LA best on **{n_la}/{len(full_rows)}** dataset x metric cells\n'
               f'* CF-Uniform best on **{n_uni}/{len(full_rows)}**\n')
    neg = [r for r in full_rows if float(r['improv_vs_best_pct']) < 0]
    rep.append(f'* cells where CF-LA is *worse* than the best non-CF-LA method: '
               f'**{len(neg)}/{len(full_rows)}**')
    for r in neg:
        rep.append(f'  * {DISPLAY_DS[r["dataset"]]} {METRIC_TEX[r["metric"]]}: '
                   f'CF-LA {float(r["cf_la"]):.6f} vs '
                   f'{r["best_non_cfla_method"]} {float(r["best_non_cfla"]):.6f} '
                   f'({float(r["improv_vs_best_pct"]):+.2f}%)')
    rep.append('')
    rep.append('## Files\n')
    for f in ('TABLE1_FINAL.tex', 'TABLE1_FINAL_FULL_PRECISION.csv',
              'TABLE1_FINAL_DISPLAY.csv', 'TABLE1_SOURCE_AUDIT.csv',
              'TABLE1_BEST_SINGLE_AUDIT.csv'):
        rep.append(f'* `{f}`')
    rep.append('')
    rep.append('## Sources\n')
    for p in (SINGLES_A, SINGLES_B, M5):
        rep.append(f'* `{p.relative_to(REPO)}` sha256 `{sha(p)}`')
    rep.append('')
    (OUT / 'TABLE1_FINAL_REPORT.md').write_text('\n'.join(rep) + '\n',
                                                encoding='utf-8')

    print(f'wrote Table 1 into {OUT}')
    print(f'  cells: {len(full_rows)}  (CF-LA overall best {n_la}, '
          f'CF-Uniform {n_uni}, CF-LA worse than best non-CF-LA {len(neg)})')
    for ds in DATASETS:
        for metric in ('NDCG@20',):
            r = by_ds[ds][metric]
            print(f'  {DISPLAY_DS[ds]:12s} {METRIC_TEX[metric]}: '
                  f'uni={float(r["cf_uniform"]):.9f} la={float(r["cf_la"]):.9f} '
                  f'imp_vs_best={float(r["improv_vs_best_pct"]):+.3f}%')


if __name__ == '__main__':
    main()
