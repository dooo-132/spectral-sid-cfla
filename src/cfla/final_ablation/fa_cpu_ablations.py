#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The CPU-only ablations of the final candidate-likelihood suite.

None of these touch a GPU: every number is recomputed from frozen candidate union/NLL
parquet, which is exactly what section 58 permits (teacher-forced evaluation is already
done; this is reranking + statistics).

Unified target handling.  The two regimes store the target differently:
  development  `cand_target` is a global SID row index  -> find it inside the segment
  Sports       `tgt_rel`     is a within-segment offset -> use it directly
`target_index()` returns a within-segment index for both, so every downstream metric
(group, rank, rescue/damage) is written once.

Subcommands
  calibration   ABLATION A   raw vs z-calibrated NLL          -> calibration_ablation.csv
  aggregation   ABLATION B   Min / Avg / LSE / Max            -> aggregation_ablation.csv
  retrieval     ABLATION C   retrieval-only vs likelihood-only vs hybrid
  beam          ABLATION F   beam budget {5,10,20,30} + count-matched control
  subgroup      ABLATION H   Only-F1 / Only-F2 / Both / Neither
  rescue        ABLATION H   rank movement and rescue/damage
  bootstrap     section 55   paired user-level bootstrap, 5000 resamples
  sports        section 53   frozen replay on Sports (development-selected settings only)
  temperature   section 18   VALID-only temperature diagnostic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fa_common as C  # noqa: E402

KS = (5, 10, 20)
PRIMARY = 20


# ---------------------------------------------------------------------------
# unified regime loading
# ---------------------------------------------------------------------------
def target_index(g, tgt, kind):
    """Within-segment index of each user's target, or -1 when absent.

    This is the ONLY place the target is located; it is never used to build a score.
    """
    n = len(g['users'])
    idx = np.full(n, -1, dtype=np.int64)
    if kind == 'rel':
        for u in range(n):
            r = int(tgt[u])
            if r >= 0:
                idx[u] = r
        return idx
    for u in range(n):
        s, e = int(g['starts'][u]), int(g['ends'][u])
        t = int(tgt[u])
        if t < 0:
            continue
        hit = np.where(g['item'][s:e] == t)[0]
        if len(hit):
            idx[u] = int(hit[0])
    return idx


def rank_at(g, scores, idx):
    """1-based rank of the located target inside its segment; -1 when absent."""
    n = len(g['users'])
    rank = np.full(n, -1, dtype=np.int64)
    for u in range(n):
        r = idx[u]
        if r < 0:
            continue
        s, e = int(g['starts'][u]), int(g['ends'][u])
        seg = scores[s:e]
        rank[u] = 1 + int((seg > seg[r]).sum())
    return rank


def load_dev(ds, split):
    u, n = C.load_cells_dev(ds, split)
    return C.grouped_arrays(u, n), C.load_targets_dev(ds, split), 'abs'


def load_sports(split):
    u, n = C.load_cells_sports(split)
    return C.grouped_arrays_asis(u, n), C.load_targets_sports_test(), 'rel'


def dev_scaler(ds):
    return C.scaler_from_json(C.CANON_SCALER, datasets_key=ds)


def sports_scaler():
    return C.scaler_from_json(C.SPORTS / 'valid' / 'candidate_nll_scaler.json')


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------
def metric_row(dataset, regime, method, rank, extra=None):
    s = C.summarize(rank, KS)
    row = {'dataset': dataset, 'regime': regime, 'method': method,
           'n_users': s['n_users'], 'coverage': s['coverage']}
    for k in KS:
        row[f'NDCG@{k}'] = s[f'NDCG@{k}']
        row[f'Recall@{k}'] = s[f'Recall@{k}']
    if extra:
        row.update(extra)
    return row


def rank_vecs(g, idx, sc, methods):
    """method -> per-user NDCG@20 vector (for bootstrap) and rank vector."""
    out = {}
    for name, spec in methods.items():
        scores = C.score(g, spec[0], sc, **spec[1]) if len(spec) > 1 else C.score(g, spec[0], sc)
        r = rank_at(g, scores, idx)
        out[name] = r
    return out


# ---------------------------------------------------------------------------
# ABLATION A — calibration
# ---------------------------------------------------------------------------
A_METHODS = [('A0_Uniform', ('Uniform', {})),
             ('A1_RawLSE', ('RawLSE', {})),
             ('A2_MeanLSE', ('MeanLSE', {})),
             ('A3_ZLSE', ('ZLSE', {})),
             ('A4_CenteredLSE', ('CenteredLSE', {}))]


def all_regimes(include_sports=True):
    for ds in C.DATASETS:
        yield ds, 'development', (lambda d=ds: load_dev(d, 'test'))
    if include_sports:
        yield C.SPORTS_DS, 'confirmatory', (lambda: load_sports('test'))


def run_calibration():
    rows, cntrl = [], []
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        C.assert_targets_align(g, tgt, n_items=C.N_ITEMS[ds])
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        rv = rank_vecs(g, idx, sc, {n: s for n, s in A_METHODS})
        run = C.make_run('calibration', ds, {'methods': [n for n, _ in A_METHODS],
                                             'split': 'test', 'regime': regime})
        for name, _ in A_METHODS:
            rows.append(metric_row(ds, regime, name, rv[name]))
        # is MeanLSE exactly monotone-equivalent to RawLSE? (section 9)
        u, n_ = (C.load_cells_dev(ds, 'test') if ds != C.SPORTS_DS
                 else C.load_cells_sports('test'))
        g2 = C.grouped_arrays(u, n_) if ds != C.SPORTS_DS else C.grouped_arrays_asis(u, n_)
        raw = C.score(g2, 'RawLSE', sc); mean = C.score(g2, 'MeanLSE', sc)
        # L=4 for both, so mean = raw/4 exactly -> identical ordering by construction
        cntrl.append({'dataset': ds, 'regime': regime,
                      'raw_vs_mean_max_abs_diff': float(np.max(np.abs(raw / C.SID_LEN - mean))),
                      'raw_vs_mean_spearman_identical':
                          bool(np.array_equal(np.argsort(np.argsort(raw)),
                                              np.argsort(np.argsort(mean))))})
        C.write_csv(run / 'results.csv', [r for r in rows if r['dataset'] == ds])
        C.finish_run(run, 'completed', f'calibration ablation, {len(A_METHODS)} methods')
        del u, n_, g, g2
    C.write_csv(C.OUT / 'calibration_ablation.csv', rows)
    C.write_csv(C.OUT / 'calibration_equivalence.csv', cntrl)
    print(f'wrote calibration_ablation.csv ({len(rows)} rows)')
    return rows, cntrl


# ---------------------------------------------------------------------------
# ABLATION B — aggregation
# ---------------------------------------------------------------------------
B_METHODS = [('B1_MinNLL', ('MinNLL', {})),
             ('B2_AvgNLL', ('AvgNLL', {})),
             ('B3_ZLSE', ('ZLSE', {})),
             ('B4_WorstNLL', ('WorstNLL', {})),
             ('A0_Uniform', ('Uniform', {}))]


def run_aggregation():
    rows = []
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        rv = rank_vecs(g, idx, sc, {n: s for n, s in B_METHODS})
        run = C.make_run('aggregation', ds, {'methods': [n for n, _ in B_METHODS],
                                             'split': 'test', 'regime': regime})
        for name, _ in B_METHODS:
            rows.append(metric_row(ds, regime, name, rv[name]))
        C.write_csv(run / 'results.csv', [r for r in rows if r['dataset'] == ds])
        C.finish_run(run, 'completed', 'aggregation ablation')
        del g
    C.write_csv(C.OUT / 'aggregation_ablation.csv', rows)
    print(f'wrote aggregation_ablation.csv ({len(rows)} rows)')
    return rows


# ---------------------------------------------------------------------------
# ABLATION C — retrieval vs likelihood
# ---------------------------------------------------------------------------
def select_lambda_dev(ds):
    """Pick lambda on VALID only, over the frozen grid, by VALID NDCG@20."""
    g, tgt, kind = load_dev(ds, 'valid')
    idx = target_index(g, tgt, kind)
    sc = dev_scaler(ds)
    best, bestv = None, -1.0
    grid = {}
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = rank_at(g, C.score(g, 'Hybrid', sc, lam=lam), idx)
        v = float(C.ndcg_from_rank(r, PRIMARY).mean())
        grid[lam] = v
        if v > bestv:
            best, bestv = lam, v
    del g
    return best, bestv, grid


def run_retrieval():
    rows, sel = [], []
    lam_dev = {}
    for ds in C.DATASETS:
        lam, v, grid = select_lambda_dev(ds)
        lam_dev[ds] = lam
        sel.append({'dataset': ds, 'selected_lambda': lam, 'valid_NDCG@20': v,
                    **{f'valid_lam_{k}': vv for k, vv in grid.items()}})
    # Sports must use a DEVELOPMENT-selected setting (section 53), never its own VALID.
    # Rule: the single global lambda maximising the MEAN valid NDCG@20 across the four
    # development datasets -- fixed before any Sports number is examined.
    grid_keys = [k for k in sel[0] if k.startswith('valid_lam_')]
    mean_by_lam = {k: float(np.mean([r[k] for r in sel])) for k in grid_keys}
    best_key = max(mean_by_lam, key=mean_by_lam.get)
    lam_sports = float(best_key.replace('valid_lam_', ''))
    sel.append({'dataset': C.SPORTS_DS, 'selected_lambda': lam_sports,
                'valid_NDCG@20': mean_by_lam[best_key],
                'note': 'development-selected global lambda (max mean VALID NDCG@20 over '
                        'the four dev datasets), frozen before any Sports number'})
    C.write_csv(C.OUT / 'retrieval_lambda_selection.csv', sel)

    for ds, regime, loader in all_regimes():
        lam = lam_sports if ds == C.SPORTS_DS else lam_dev[ds]
        g, tgt, kind = loader()
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        methods = {
            'C1_RetrievalOnly_Uniform': C.score(g, 'Uniform', sc),
            'C2_LikelihoodOnly_ZLSE': C.score(g, 'ZLSE', sc),
            'C3_HybridFixed_lam0.5': C.score(g, 'Hybrid', sc, lam=0.5),
            'C4_HybridValidLambda': C.score(g, 'Hybrid', sc, lam=lam),
        }
        run = C.make_run('retrieval', ds, {'methods': list(methods), 'lambda': lam,
                                           'split': 'test', 'regime': regime})
        for name, scv in methods.items():
            rows.append(metric_row(ds, regime, name, rank_at(g, scv, idx),
                                   {'lambda': lam if 'Hybrid' in name else ''}))
        C.write_csv(run / 'results.csv', [r for r in rows if r['dataset'] == ds])
        C.finish_run(run, 'completed', 'retrieval vs likelihood ablation')
        del g
    C.write_csv(C.OUT / 'retrieval_likelihood_ablation.csv', rows)
    print(f'wrote retrieval_likelihood_ablation.csv ({len(rows)} rows)')
    return rows


# ---------------------------------------------------------------------------
# ABLATION F — beam / candidate budget
# ---------------------------------------------------------------------------
def beam_mask(g, B):
    """Candidates surviving a beam-B regime: present in either factorization's top-B."""
    r1 = g['rank1']; r2 = g['rank2']
    in1 = (r1 >= 1) & (r1 <= B)
    in2 = (r2 >= 1) & (r2 <= B)
    return in1 | in2


def count_matched_keep(g, B, scores):
    """Per user, keep the top-B candidates by `scores`, matching |C| across methods."""
    keep = np.zeros(len(scores), dtype=bool)
    for s, e in zip(g['starts'], g['ends']):
        seg = scores[s:e]
        k = min(B, len(seg))
        top = np.argpartition(-seg, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        keep[s + top] = True
    return keep


def run_beam():
    rows, cntrl = [], []
    budgets = (5, 10, 20, 30)
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        for B in budgets:
            m = beam_mask(g, B)
            idxs = np.where(m)[0]
            # user id per surviving row, in original order
            seg_user = np.repeat(g['users'], g['ends'] - g['starts'])
            su = seg_user[idxs]
            users2, st2 = np.unique(su, return_index=True)
            en2 = np.append(st2[1:], len(idxs))
            gb = {'users': users2, 'starts': st2, 'ends': en2}
            gb['item'] = g['item'][idxs]
            for k in ('p1', 'p2', 'N1', 'N2', 'rank1', 'rank2', 'in1', 'in2'):
                gb[k] = g[k][idxs]
            # target index within the truncated segment
            idx_full = target_index(g, tgt, kind)   # WITHIN-segment index
            # map to surviving rows
            old_to_new = -np.ones(len(g['item']), dtype=np.int64)
            old_to_new[idxs] = np.arange(len(idxs))
            idxb = np.full(len(users2), -1, dtype=np.int64)
            # user -> position in users2
            upos = {int(u): j for j, u in enumerate(users2)}
            for i in range(len(g['users'])):
                if idx_full[i] < 0:
                    continue
                j = upos.get(int(g['users'][i]))
                if j is None:
                    continue
                # global row = segment start + within-segment index.  Using idx_full
                # directly indexes the global array with a local offset, which silently
                # zeroes all coverage -- that bug invalidated the first beam run.
                newpos = old_to_new[int(g['starts'][i]) + int(idx_full[i])]
                if newpos >= 0 and st2[j] <= newpos < en2[j]:
                    idxb[j] = newpos - st2[j]

            union_size = float(np.mean(en2 - st2))
            # the frozen scaler is reused unchanged; it is NOT refit on a truncated
            # union, which would make beam budgets incomparable
            uni = C.score(gb, 'Uniform', sc); lse = C.score(gb, 'ZLSE', sc)
            r_uni = rank_at(gb, uni, idxb); r_lse = rank_at(gb, lse, idxb)
            row = {'dataset': ds, 'regime': regime, 'beam_B': B,
                   'mean_union_size': union_size,
                   'coverage': float((idxb >= 0).mean()),
                   'oracle_coverage': float((idxb >= 0).mean()),
                   'jaccard_F1_F2': float(((g['rank1'] >= 1) & (g['rank1'] <= B) &
                                           (g['rank2'] >= 1) & (g['rank2'] <= B)).sum() /
                                          max(1, ((g['rank1'] >= 1) & (g['rank1'] <= B)).sum() +
                                              ((g['rank2'] >= 1) & (g['rank2'] <= B)).sum() -
                                              ((g['rank1'] >= 1) & (g['rank1'] <= B) &
                                               (g['rank2'] >= 1) & (g['rank2'] <= B)).sum()))}
            for nm, rr in (('Uniform', r_uni), ('ZLSE', r_lse)):
                s = C.summarize(rr, KS)
                row[f'{nm}_NDCG@20'] = s['NDCG@20']
                row[f'{nm}_Recall@20'] = s['Recall@20']
            row['Oracle_NDCG@20'] = float(C.ndcg_from_rank(
                np.where(idxb >= 0, 1, -1), PRIMARY).mean())
            rows.append(row)

            # count-matched control: |C| equal across methods (section 42)
            for cap in (10, 20, 30):
                ku = count_matched_keep(gb, cap, uni)
                kk = count_matched_keep(gb, cap, lse)
                gsub = dict(gb)
                m2 = ku & kk
                idxs2 = np.where(m2)[0]
                if len(idxs2) == 0:
                    continue
                su2 = np.repeat(gb['users'], gb['ends'] - gb['starts'])[idxs2]
                u3, s3 = np.unique(su2, return_index=True)
                e3 = np.append(s3[1:], len(idxs2))
                for k in ('item', 'p1', 'p2', 'N1', 'N2', 'rank1', 'rank2', 'in1', 'in2'):
                    gsub[k] = gb[k][idxs2]
                gsub['users'], gsub['starts'], gsub['ends'] = u3, s3, e3
                o2n = -np.ones(len(gb['item']), dtype=np.int64); o2n[idxs2] = np.arange(len(idxs2))
                ib = np.full(len(u3), -1, dtype=np.int64)
                up = {int(u): j for j, u in enumerate(u3)}
                for i in range(len(gb['users'])):
                    if idxb[i] < 0:
                        continue
                    j = up.get(int(gb['users'][i]))
                    if j is None:
                        continue
                    np_ = o2n[gb['starts'][i] + idxb[i]]
                    if np_ >= 0 and s3[j] <= np_ < e3[j]:
                        ib[j] = np_ - s3[j]
                ru = rank_at(gsub, C.score(gsub, 'Uniform', sc), ib)
                rl = rank_at(gsub, C.score(gsub, 'ZLSE', sc), ib)
                cntrl.append({'dataset': ds, 'regime': regime, 'beam_B': B,
                              'cap': cap, 'mean_union_size': float(np.mean(e3 - s3)),
                              'Uniform_NDCG@20': float(C.ndcg_from_rank(ru, PRIMARY).mean()),
                              'ZLSE_NDCG@20': float(C.ndcg_from_rank(rl, PRIMARY).mean()),
                              'gain': float(C.ndcg_from_rank(rl, PRIMARY).mean()
                                            - C.ndcg_from_rank(ru, PRIMARY).mean())})
            del gb
        run = C.make_run('beam_budget', ds, {'budgets': list(budgets),
                                             'split': 'test', 'regime': regime,
                                             'code_rev': 'r2-target-remap-fix'})
        C.write_csv(run / 'results.csv', [r for r in rows if r['dataset'] == ds])
        C.finish_run(run, 'completed', 'beam budget + count-matched control')
        del g
    C.write_csv(C.OUT / 'beam_budget_ablation.csv', rows)
    C.write_csv(C.OUT / 'beam_budget_count_matched.csv', cntrl)
    print(f'wrote beam_budget_ablation.csv ({len(rows)} rows) '
          f'+ count-matched ({len(cntrl)} rows)')
    return rows, cntrl


# ---------------------------------------------------------------------------
# ABLATION H — subgroup, rank movement, rescue/damage
# ---------------------------------------------------------------------------
def groups_at(g, idx, k=20):
    n = len(g['users'])
    out = np.full(n, 'Neither', dtype=object)
    for u in range(n):
        r = idx[u]
        if r < 0:
            continue
        s = int(g['starts'][u])
        h1 = 1 <= int(g['rank1'][s + r]) <= k
        h2 = 1 <= int(g['rank2'][s + r]) <= k
        out[u] = 'Both' if (h1 and h2) else ('Only_F1' if h1 else ('Only_F2' if h2 else 'Neither'))
    return out


def run_subgroup():
    rows = []
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        grp = groups_at(g, idx, PRIMARY)
        r_uni = rank_at(g, C.score(g, 'Uniform', sc), idx)
        r_lse = rank_at(g, C.score(g, 'ZLSE', sc), idx)
        r_min = rank_at(g, C.score(g, 'MinNLL', sc), idx)
        for grpname in ('Only_F1', 'Only_F2', 'Both', 'Neither'):
            m = grp == grpname
            if m.sum() == 0:
                continue
            for nm, rr in (('Uniform', r_uni), ('ZLSE', r_lse), ('MinNLL', r_min)):
                s = C.summarize(rr[m], KS)
                rows.append({'dataset': ds, 'regime': regime, 'group': grpname,
                             'method': nm, 'n': int(m.sum()), **s})
    C.write_csv(C.OUT / 'subgroup_analysis.csv', rows)
    print(f'wrote subgroup_analysis.csv ({len(rows)} rows)')
    return rows


def run_rescue():
    rows, move = [], []
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        r_uni = rank_at(g, C.score(g, 'Uniform', sc), idx)
        r_lse = rank_at(g, C.score(g, 'ZLSE', sc), idx)
        in_union = idx >= 0
        for k in (5, 10, 20):
            rescue = in_union & (r_uni > k) & (r_lse >= 1) & (r_lse <= k)
            damage = in_union & (r_uni >= 1) & (r_uni <= k) & ((r_lse > k) | (r_lse < 0))
            denom_r = int((in_union & (r_uni > k)).sum())
            denom_d = int((in_union & (r_uni >= 1) & (r_uni <= k)).sum())
            rows.append({'dataset': ds, 'regime': regime, 'k': k,
                         'n_target_in_union': int(in_union.sum()),
                         'rescue': int(rescue.sum()),
                         'rescue_rate': float(rescue.sum() / denom_r) if denom_r else 0.0,
                         'damage': int(damage.sum()),
                         'damage_rate': float(damage.sum() / denom_d) if denom_d else 0.0})
        both = in_union & (r_uni >= 1) & (r_lse >= 1)
        delta = np.where(both, r_uni - r_lse, 0)
        moved = delta != 0
        move.append({'dataset': ds, 'regime': regime,
                     'n_target_in_union': int(in_union.sum()),
                     'improved': int((delta > 0).sum()), 'unchanged': int((delta == 0).sum()),
                     'worsened': int((delta < 0).sum()),
                     'mean_rank_uniform': float(r_uni[both].mean()) if both.any() else float('nan'),
                     'mean_rank_lse': float(r_lse[both].mean()) if both.any() else float('nan'),
                     'mean_rank_delta': float(delta[moved].mean()) if moved.any() else 0.0})
    C.write_csv(C.OUT / 'rescue_damage.csv', rows)
    C.write_csv(C.OUT / 'rank_movement.csv', move)
    print(f'wrote rescue_damage.csv ({len(rows)} rows) + rank_movement.csv')
    return rows, move


# ---------------------------------------------------------------------------
# section 55 — bootstrap
# ---------------------------------------------------------------------------
COMPARISONS = [
    ('LSE_vs_Uniform', 'ZLSE', 'Uniform'),
    ('zLSE_vs_RawLSE', 'ZLSE', 'RawLSE'),
    ('zLSE_vs_MeanLSE', 'ZLSE', 'MeanLSE'),
    ('zLSE_vs_CenteredLSE', 'ZLSE', 'CenteredLSE'),
    ('LSE_vs_MinNLL', 'ZLSE', 'MinNLL'),
    ('LSE_vs_AvgNLL', 'ZLSE', 'AvgNLL'),
    ('LSE_vs_WorstNLL', 'ZLSE', 'WorstNLL'),
    ('MinNLL_vs_Uniform', 'MinNLL', 'Uniform'),
    ('WorstNLL_vs_Uniform', 'WorstNLL', 'Uniform'),
    ('LSE_vs_HybridFixed', 'ZLSE', 'HybridFixed'),
]


def run_bootstrap():
    rows = []
    for ds, regime, loader in all_regimes():
        g, tgt, kind = loader()
        idx = target_index(g, tgt, kind)
        sc = sports_scaler() if ds == C.SPORTS_DS else dev_scaler(ds)
        vec = {}
        for nm, spec in (('Uniform', 'Uniform'), ('RawLSE', 'RawLSE'),
                         ('MeanLSE', 'MeanLSE'), ('CenteredLSE', 'CenteredLSE'),
                         ('ZLSE', 'ZLSE'), ('MinNLL', 'MinNLL'), ('AvgNLL', 'AvgNLL'),
                         ('WorstNLL', 'WorstNLL'),
                         ('HybridFixed', 'Hybrid')):
            r = rank_at(g, C.score(g, spec, sc, lam=0.5), idx)
            vec[nm] = C.ndcg_from_rank(r, PRIMARY)
        for label, a, b in COMPARISONS:
            st = C.paired_bootstrap(vec[a], vec[b])
            rows.append({'comparison': label, 'dataset': ds, 'regime': regime,
                         'metric': f'NDCG@{PRIMARY}', **st})
    C.write_csv(C.OUT / 'bootstrap.csv', rows)
    print(f'wrote bootstrap.csv ({len(rows)} rows)')
    return rows


# ---------------------------------------------------------------------------
# section 18 — temperature diagnostic (VALID only)
# ---------------------------------------------------------------------------
def run_temperature():
    rows = []
    for ds in C.DATASETS:
        g, tgt, kind = load_dev(ds, 'valid')
        idx = target_index(g, tgt, kind)
        sc = dev_scaler(ds)
        for T in (0.5, 1.0, 2.0):
            r = rank_at(g, C.score(g, 'LSE_T', sc, T=T), idx)
            rows.append({'dataset': ds, 'split': 'valid', 'T': T,
                         'valid_NDCG@20': float(C.ndcg_from_rank(r, PRIMARY).mean()),
                         'valid_Recall@20': float(C.recall_from_rank(r, PRIMARY).mean())})
        del g
    C.write_csv(C.OUT / 'temperature_diagnostic.csv', rows)
    print(f'wrote temperature_diagnostic.csv ({len(rows)} rows)')
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['calibration', 'aggregation', 'retrieval', 'beam',
                                    'subgroup', 'rescue', 'bootstrap', 'temperature'])
    a = ap.parse_args(argv)
    C.OUT.mkdir(parents=True, exist_ok=True)
    return {'calibration': lambda: (run_calibration(), 0)[1],
            'aggregation': lambda: (run_aggregation(), 0)[1],
            'retrieval': lambda: (run_retrieval(), 0)[1],
            'beam': lambda: (run_beam(), 0)[1],
            'subgroup': lambda: (run_subgroup(), 0)[1],
            'rescue': lambda: (run_rescue(), 0)[1],
            'bootstrap': lambda: (run_bootstrap(), 0)[1],
            'temperature': lambda: (run_temperature(), 0)[1]}[a.cmd]()


if __name__ == '__main__':
    sys.exit(main())
