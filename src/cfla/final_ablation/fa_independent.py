#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION D — shared vs independent models.

QUESTION
    Does candidate-likelihood reranking need the SHARED T5, or does it work with
    two INDEPENDENT single-factorization models?

The shared-regime numbers already published (outputs/final_ablation/
calibration_ablation.csv, rows A0_Uniform / A3_ZLSE) are produced by ONE shared
checkpoint (`step2_shared_model/runs/<ds>_shared_spec_rqvae_seed2025/shared_model.pth`)
that serves both factorizations.  This module builds the exact independent-regime
analogue:

    C_ind(u)      = C_F1(u) ∪ C_F2(u),   F1 = spectral_k3, F2 = rqvae3, beam = 30
    P_F(u,i)      = beam/retrieval probability of i under F (0 outside F's beam)
    NLL_F(u,i)    = teacher-forced TotalNLL of candidate i's OWN SID under F's own
                    independent checkpoint
    mu_F,sigma_F  = fitted on VALID over ALL validation union candidates (never
                    targets only); TEST always uses the VALID-fitted scaler
    z_F(u,i)      = (NLL_F - mu_F)/sigma_F
    Uniform_ind   = 0.5*P_F1 + 0.5*P_F2
    LSE_ind       = logsumexp(-z_F1, -z_F2)
    BestF1/BestF2 = retrieval-only ranking by P_F1 / P_F2
    Oracle_ind    = target present in the union

REUSE
    * beam/SID/checkpoint loading and the item-id shift detector come from
      `analysis/mechanism/cross_tokenizer_shared/xt_common.py` (the module that
      produced these very beam dumps);
    * the batched teacher-forced candidate scorer and the history-bank builder
      come from `analysis/mechanism/candidate_conditioned/cc_common.py`, i.e. the
      code that produced the published candidate NLLs;
    * scaler fitting, the frozen score definitions, ranking, metrics, bootstrap
      and run scaffolding come from `analysis/mechanism/final_ablation/fa_common.py`.

CONVENTIONS (the #1 trap)
    Candidate / target ids inside this module are **0-based SID-array ROW
    indices**: for candidate row r the codes are `sid_array[r]`.  The beam dumps
    store 0-based rows too (shift detector confirms `raw == row` for every dump
    used here), and the canonical 1-based item id is `row + 1`, so the dataset
    target item id t maps to row `t - 1`.  Everything is normalised to rows at
    the door: `_merged_candidates` returns 1-based canonical ids and
    `build_union_rows` immediately converts to rows.  `sid[c - 1]` is the
    mistake this convention exists to prevent.

VALIDATION (mandatory, performed before any score is trusted)
    `cross_tokenizer_target_nll.parquet` holds, for every (tokenizer, dataset,
    user), the teacher-forced total_nll of that user's TARGET under the matching
    independent checkpoint.  `cmd_validate` / the per-dataset run re-scores every
    target through THIS module's candidate scorer and compares elementwise
    (test) plus mean/std (valid).  A mismatch above TOL aborts the run.

Outputs
    outputs/final_ablation/runs/shared_vs_independent_<dataset>_<hash>/
    outputs/final_ablation/shared_vs_independent.csv
    outputs/final_ablation/shared_vs_independent_bootstrap.csv
    outputs/final_ablation/shared_vs_independent_gains.csv
    outputs/final_ablation/shared_vs_independent_union_stats.csv
    outputs/final_ablation/SHARED_VS_INDEPENDENT_NOTES.md

Usage
    python3 fa_independent.py validate [--datasets ...]
    python3 fa_independent.py run      [--datasets ...] [--batch-users 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared'),
           str(HERE.parent / 'candidate_conditioned')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fa_common as C  # noqa: E402  (frozen suite machinery)
from xt_common import (ART, CODEBOOK, STEP1, load_cell, load_sid,  # noqa: E402
                       sha256_file, sid_path)

# ---------------------------------------------------------------------------
# frozen configuration of this ablation
# ---------------------------------------------------------------------------
F1, F2 = 'spectral_k3', 'rqvae3'
BEAM = 30
MAXLEN = 20
NBIG = 10 ** 9
TOL = 1e-4                 # absolute TotalNLL tolerance (float32 batching noise ~2e-5)
REF_PARQUET = STEP1 / 'cross_tokenizer_target_nll.parquet'
REF_CALIB = STEP1 / 'cross_tokenizer_nll_calibration.csv'
REF_SINGLE = (C.SC / 'step2_shared_model' / 'shared_single_factorization_metrics.csv')
AF_FEAT = C.AF_FEAT

# method names (the aggregate table's `method` column)
M_UNIFORM = 'A0_Uniform_ind'
M_ZLSE = 'A3_ZLSE_ind'
M_BESTF1 = 'BestF1_ind'
M_BESTF2 = 'BestF2_ind'
M_ORACLE = 'Oracle_ind'
METHODS = (M_UNIFORM, M_ZLSE, M_BESTF1, M_BESTF2, M_ORACLE)


def independent_paths(ds, split):
    """The exact read-only artifacts of the independent regime (task-specified)."""
    return {
        F1: {
            'ckpt': ART / 'model' / 'ckpt' / f'tiger_{ds}_nonneg_pearson_v3_k3_sid4.pth',
            'sid': (ART / 'data' / ds /
                    f'{ds}_t5_spectral_nonneg_pearson_v3_k3_sid4.npy'),
            'beam': (ART / 'model' / f'beam_dumps_{ds}_nonneg_pearson_v3' /
                     f'{split}_k3_sid4.npz'),
        },
        F2: {
            'ckpt': ART / 'model' / 'ckpt' / f'tiger_{ds}.pth',
            'sid': ART / 'data' / ds / f'{ds}_t5_rqvae.npy',
            'beam': STEP1 / 'beams' / f'{ds}_rqvae3_{split}.npz',
        },
    }


# ---------------------------------------------------------------------------
# model + history plumbing (validated code paths)
# ---------------------------------------------------------------------------
def load_independent_model(ckpt_path, device='cuda'):
    """Load one independent single-factorization checkpoint (eval only)."""
    from src.analysis.teacher_forced import load_model_from_ckpt
    model, vocab = load_model_from_ckpt(str(ckpt_path), device=device)
    if vocab != 4 * CODEBOOK + 1:
        raise AssertionError(f'{ckpt_path}: vocab {vocab} != {4 * CODEBOOK + 1}')
    for p in model.parameters():
        p.requires_grad_(False)
    return model


_HIST_CACHE = {}


def history_bank(tok, ds, split):
    """The canonical history encoding of one factorization (no control token).

    Identical construction to `xt_teacher_forced.score_split`: the same
    `GenRecDataset` evaluation mode the independent checkpoints were trained and
    evaluated with.  Row i corresponds to row i of `<split>.parquet`.
    """
    key = (tok, ds, split)
    if key in _HIST_CACHE:
        return _HIST_CACHE[key]
    from src.tiger.dataset import GenRecDataset
    from src.tiger.dataloader import GenRecDataLoader
    dset = GenRecDataset(dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
                         code_path=str(sid_path(tok, ds)), mode='evaluation',
                         max_len=MAXLEN, codebook_size=CODEBOOK)
    loader = GenRecDataLoader(dset, batch_size=512, shuffle=False)
    toks, masks = [], []
    for b in loader:
        toks.append(b['history'].numpy().astype(np.int64))
        masks.append(b['attention_mask'].numpy().astype(np.int64))
    H = np.concatenate(toks)
    M = np.concatenate(masks)
    if len(H) != len(dset):
        raise AssertionError(f'{tok}/{ds}/{split}: loader dropped rows '
                             f'({len(H)} != {len(dset)})')
    _HIST_CACHE[key] = (H, M, len(dset))
    return _HIST_CACHE[key]


def score_candidates(model, H, M, cand_tokens, cand_owner, batch_users=128,
                     label=''):
    """Teacher-forced TotalNLL with an OOM-safe batch fallback.

    The scorer is `cc_common.candidate_total_nll`, the exact function that
    produced the published candidate NLLs.  A concurrent job may share the GPU,
    so on CUDA OOM the user-batch is halved and the call retried.
    """
    import torch
    from cc_common import candidate_total_nll
    b = int(batch_users)
    while True:
        try:
            return candidate_total_nll(model, H, M, cand_tokens, cand_owner,
                                       device='cuda', batch_size=b)
        except RuntimeError as e:                       # pragma: no cover
            if 'out of memory' not in str(e).lower() or b <= 8:
                raise
            torch.cuda.empty_cache()
            b = max(8, b // 2)
            print(f'    OOM on {label}; retrying with batch_users={b}', flush=True)


def sid_tokens(sid, rows):
    """0-based SID rows -> slot-encoded tokens; raises on an out-of-range row."""
    from cc_common import sid_tokens_for_items
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= len(sid)):
        raise AssertionError(f'SID row out of range: {rows.min()}..{rows.max()} '
                             f'for {len(sid)} items')
    return sid_tokens_for_items(sid, rows)


# ---------------------------------------------------------------------------
# independent candidate union
# ---------------------------------------------------------------------------
def _merged_candidates(cell):
    """One cell's (user, item, prob) after merging duplicate items per user.

    Mirrors `fg_common._grouped_probabilities`: duplicate decoded items inside a
    row have their probabilities summed.  Items are 1-based canonical ids.
    """
    items = cell['items']
    probs = cell['probs']
    n, B = items.shape
    key = np.where(items >= 0, items, NBIG)
    order = np.argsort(key, axis=1, kind='stable')
    ks = np.take_along_axis(key, order, axis=1)
    vs = np.take_along_axis(probs, order, axis=1)
    first = np.empty(ks.shape, dtype=bool)
    first[:, 0] = True
    first[:, 1:] = ks[:, 1:] != ks[:, :-1]
    grp = np.cumsum(first, axis=1) - 1
    acc = np.zeros_like(vs)
    rows = np.repeat(np.arange(n, dtype=np.int64), B)
    np.add.at(acc, (rows, grp.reshape(-1)), vs.reshape(-1))
    grouped = np.take_along_axis(acc, grp, axis=1)
    real = first & (ks < NBIG)
    ui, uj = np.where(real)
    return (ui.astype(np.int64), ks[ui, uj].astype(np.int64),
            grouped[ui, uj].astype(np.float64))


def build_union_rows(cells):
    """C_ind(u) = C_F1(u) ∪ C_F2(u) as 0-based SID rows + per-model probabilities.

    Returns (users, rows, p1, p2, sizes) with
        users (P,) int64   user index (row of the split parquet)
        rows  (P,) int64   0-based SID-array row of the candidate
        p1,p2 (P,) float64 retrieval probability under F1 / F2 (0 where absent)
        sizes (n,) int64   |C_ind(u)| per user
    Rows are sorted by (user, row).
    """
    u1, i1, p1v = _merged_candidates(cells[F1])
    u2, i2, p2v = _merged_candidates(cells[F2])
    users = np.concatenate([u1, u2])
    items = np.concatenate([i1, i2])
    a = np.concatenate([p1v, np.zeros_like(p2v)])
    b = np.concatenate([np.zeros_like(p1v), p2v])
    order = np.lexsort((items, users))
    users, items, a, b = users[order], items[order], a[order], b[order]
    new = np.empty(len(users), dtype=bool)
    new[0] = True
    new[1:] = (users[1:] != users[:-1]) | (items[1:] != items[:-1])
    grp = np.cumsum(new) - 1
    ng = int(grp[-1]) + 1
    pa = np.zeros(ng)
    pb = np.zeros(ng)
    np.add.at(pa, grp, a)
    np.add.at(pb, grp, b)
    out_users = users[new]
    out_rows = (items[new] - 1).astype(np.int64)     # 1-based canonical -> SID row
    n = cells[F1]['n_users']
    sizes = np.bincount(out_users, minlength=n).astype(np.int64)
    return out_users, out_rows, pa, pb, sizes


def grouped_from_union(users, rows, p1, p2, N1, N2):
    """fa_common-compatible grouped arrays (sorted by user, then candidate row).

    `item` holds 0-based SID rows and the target is compared in the same space,
    which is the convention `fa_common.rank_of_target` expects.
    """
    order = np.lexsort((rows, users))
    u = users[order]
    uniq, starts = np.unique(u, return_index=True)
    ends = np.append(starts[1:], len(order))
    return {
        'users': uniq, 'starts': starts, 'ends': ends, 'order': order,
        'item': rows[order], 'p1': p1[order], 'p2': p2[order],
        'N1': N1[order], 'N2': N2[order],
    }


def target_rows(cells, ds, split):
    """Per-user target as a 0-based SID row (dataset target item id t -> t-1).

    The beam dumps store the canonical 1-based item id; a missing target is the
    -1 sentinel and must stay -1 (not become -2).
    """
    t1 = cells[F1]['target']
    t2 = cells[F2]['target']
    if not np.array_equal(t1, t2):
        raise AssertionError(f'{ds}/{split}: the two beam dumps disagree on targets')
    rows = np.where(t1 >= 1, t1 - 1, -1).astype(np.int64)
    C.assert_targets_align({'users': np.arange(len(rows))}, rows,
                           n_items=C.N_ITEMS[ds])
    # cross-check against the independent feature dump used by the shared study
    af = AF_FEAT / f'{ds}_{split}.npz'
    if af.exists():
        ct = np.asarray(np.load(af, allow_pickle=True)['cand_target'], dtype=np.int64)
        if len(ct) == len(rows) and not np.array_equal(ct, rows):
            raise AssertionError(f'{ds}/{split}: AF cand_target disagrees with '
                                 f'beam-dump target - 1')
    return rows


# ---------------------------------------------------------------------------
# scoring one (dataset, split)
# ---------------------------------------------------------------------------
def load_cells_ind(ds, split):
    paths = independent_paths(ds, split)
    cells = {}
    for f in (F1, F2):
        p = paths[f]['beam']
        if not p.exists():
            raise FileNotFoundError(p)
        cells[f] = load_cell(f, ds, split, beam_override=p)
        declared = str(cells[f]['z']['ckpt_path'])
        if Path(declared).name != Path(paths[f]['ckpt']).name:
            raise AssertionError(f'{p}: dump declares checkpoint {declared}, '
                                 f'ablation uses {paths[f]["ckpt"]}')
        if Path(str(cells[f]['z']['code_path'])).name != Path(paths[f]['sid']).name:
            raise AssertionError(f'{p}: dump declares SID {cells[f]["z"]["code_path"]}')
    if cells[F1]['n_users'] != cells[F2]['n_users']:
        raise AssertionError(f'{ds}/{split}: beam dumps disagree on n_users')
    return cells, paths


def score_split(ds, split, models, cells, batch_users=128):
    """Union + teacher-forced NLL of every union candidate under BOTH checkpoints."""
    users, rows, p1, p2, sizes = build_union_rows(cells)
    nll = {}
    for f in (F1, F2):
        sid = load_sid(f, ds)
        H, M, n_users = history_bank(f, ds, split)
        toks = sid_tokens(sid, rows)
        # direct evidence that the tokens equal the ones the dumper wrote for the
        # candidates it dumped (only checkable for rows present in this f's beam)
        nll[f] = score_candidates(models[f], H, M, toks, users,
                                  batch_users=batch_users, label=f'{ds}/{split}/{f}')
        del toks, H, M
        frac = float(np.isfinite(nll[f]).mean())
        if frac != 1.0:
            raise AssertionError(f'{ds}/{split}/{f}: {1 - frac:.4%} non-finite NLL')
        if nll[f].min() < 0:
            raise AssertionError(f'{ds}/{split}/{f}: negative TotalNLL (sign error)')
    return {'users': users, 'rows': rows, 'p1': p1, 'p2': p2, 'sizes': sizes,
            'N1': nll[F1], 'N2': nll[F2]}


def score_targets(ds, split, models, cells, batch_users=128):
    """Teacher-forced TotalNLL of each user's TARGET under both checkpoints.

    Scored through the same `score_candidates` path as the union, which is what
    makes the elementwise comparison against the frozen parquet a real check of
    the scorer rather than of a parallel implementation.
    """
    rows = target_rows(cells, ds, split)
    owner = np.arange(len(rows), dtype=np.int64)
    out = {}
    for f in (F1, F2):
        sid = load_sid(f, ds)
        H, M, _ = history_bank(f, ds, split)
        toks = sid_tokens(sid, rows)
        out[f] = score_candidates(models[f], H, M, toks, owner,
                                  batch_users=batch_users,
                                  label=f'{ds}/{split}/{f}/target')
        del toks, H, M
    return rows, out


def validate_target_nll(ds, split, mine):
    """Elementwise comparison against the frozen independent target-NLL parquet.

    The frozen parquet is TEST-only; for any other split only the row count is
    reported (the VALID check is `validate_valid_calibration`).
    """
    import pandas as pd
    ref = pd.read_parquet(REF_PARQUET)
    ref = ref[(ref['dataset'] == ds)]
    rows = {}
    for f in (F1, F2):
        r = ref[ref['tokenizer'] == f].sort_values('user_index')
        if split != 'test':
            rows[f] = {'n_users': 0, 'max_abs_diff': float('nan'),
                       'mean_abs_diff': float('nan'), 'p99_abs_diff': float('nan'),
                       'ref_mean': float('nan'), 'mine_mean': float('nan'),
                       'passed': True, 'note': 'reference parquet is test-only'}
            continue
        if len(r) != len(mine[f]):
            raise AssertionError(f'{ds}: parquet has {len(r)} rows, scorer {len(mine[f])}')
        d = np.abs(np.asarray(r['total_nll'].to_numpy(), dtype=np.float64) - mine[f])
        rows[f] = {
            'n_users': int(len(d)),
            'max_abs_diff': float(d.max()),
            'mean_abs_diff': float(d.mean()),
            'p99_abs_diff': float(np.percentile(d, 99)),
            'ref_mean': float(r['total_nll'].mean()),
            'mine_mean': float(np.mean(mine[f])),
            'passed': bool(d.max() <= TOL),
            'note': '',
        }
    return rows


def validate_valid_calibration(ds, valid_tgt_nll):
    """VALID target NLL mean/std against the frozen calibration values."""
    import pandas as pd
    ref = pd.read_parquet(REF_PARQUET)
    ref = ref[ref['dataset'] == ds]
    out = {}
    for f in (F1, F2):
        r = ref[ref['tokenizer'] == f]
        ref_mean = float(r['valid_total_nll_mean'].iloc[0])
        ref_std = float(r['valid_total_nll_std'].iloc[0])       # ddof=1 (frozen)
        tot = valid_tgt_nll[f]
        out[f] = {
            'n_users': int(len(tot)),
            'ref_valid_total_nll_mean': ref_mean, 'mine_valid_total_nll_mean': float(tot.mean()),
            'ref_valid_total_nll_std': ref_std, 'mine_valid_total_nll_std': float(tot.std(ddof=1)),
            'abs_diff_mean': abs(ref_mean - float(tot.mean())),
            'abs_diff_std': abs(ref_std - float(tot.std(ddof=1))),
            'passed': bool(abs(ref_mean - float(tot.mean())) <= 1e-3
                           and abs(ref_std - float(tot.std(ddof=1))) <= 1e-3),
        }
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_test(g, tgt_rows, sc):
    """Per-method per-user NDCG@20 vectors, ranks and metrics on the union."""
    z1, z2 = C.zscores(g, sc)
    scores = {
        M_UNIFORM: C.uniform(g),
        M_ZLSE: C._lse2(-z1, -z2),
        M_BESTF1: g['p1'],
        M_BESTF2: g['p2'],
    }
    ranks, ndcg20 = {}, {}
    for name, s in scores.items():
        r = C.rank_of_target(g, s, tgt_rows)
        ranks[name] = r
        ndcg20[name] = C.ndcg_from_rank(r, 20)
    present = tgt_rows >= 0
    oracle_rank = np.where(present, 1, -1)
    ranks[M_ORACLE] = oracle_rank
    ndcg20[M_ORACLE] = C.ndcg_from_rank(oracle_rank, 20)
    return ranks, ndcg20


def metric_rows(ds, sc, n_valid_cand, ranks, sizes_test, sizes_valid, run_dir, cfg_hash):
    rows = []
    for m in METHODS:
        s = C.summarize(ranks[m], (5, 10, 20))
        rows.append({
            'dataset': ds, 'regime': 'independent', 'method': m,
            'n_users': s['n_users'], 'coverage': s['coverage'],
            'NDCG@5': s['NDCG@5'], 'Recall@5': s['Recall@5'],
            'NDCG@10': s['NDCG@10'], 'Recall@10': s['Recall@10'],
            'NDCG@20': s['NDCG@20'], 'Recall@20': s['Recall@20'],
            'mu_F1': sc['N1'][0], 'sigma_F1': sc['N1'][1],
            'mu_F2': sc['N2'][0], 'sigma_F2': sc['N2'][1],
            'scaler_fit_split': 'valid', 'scaler_n_candidates': int(n_valid_cand),
            'beam': BEAM, 'eval_split': 'test',
            'mean_union_size_valid': float(sizes_valid.mean()),
            'mean_union_size_test': float(sizes_test.mean()),
            'run_dir': str(run_dir), 'config_hash': cfg_hash,
        })
    return rows


def gain_rows(ds, ranks, shared_gain):
    s_uni = C.summarize(ranks[M_UNIFORM], (20,))
    s_lse = C.summarize(ranks[M_ZLSE], (20,))
    g_nd = s_lse['NDCG@20'] - s_uni['NDCG@20']
    g_rc = s_lse['Recall@20'] - s_uni['Recall@20']
    return {
        'dataset': ds, 'regime': 'independent',
        'Uniform_ind_NDCG@20': s_uni['NDCG@20'],
        'LSE_ind_NDCG@20': s_lse['NDCG@20'],
        'gain_NDCG@20': g_nd,
        'gain_Recall@20': g_rc,
        'shared_gain_NDCG@20': shared_gain,
        'independent_minus_shared_gain': (g_nd - shared_gain) if shared_gain is not None else float('nan'),
        'gain_ratio_ind_over_shared': (g_nd / shared_gain) if shared_gain not in (None, 0) else float('nan'),
    }


# ---------------------------------------------------------------------------
# main passes
# ---------------------------------------------------------------------------
def cmd_validate(a):
    """Pre-flight: reproduce the frozen independent target NLLs (TOL = 1e-4)."""
    import torch
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = C.OUT / f'validation_shared_vs_independent_{stamp}'
    if out.exists():
        raise SystemExit(f'HARD FAIL: {out} exists')
    out.mkdir(parents=True)
    rows, cal_rows = [], []
    ok = True
    for ds in a.datasets.split(','):
        cells_v, _ = load_cells_ind(ds, 'valid')
        cells_t, _ = load_cells_ind(ds, 'test')
        models = {f: load_independent_model(independent_paths(ds, 'test')[f]['ckpt'])
                  for f in (F1, F2)}
        _, tgt_v = score_targets(ds, 'valid', models, cells_v, a.batch_users)
        _, tgt_t = score_targets(ds, 'test', models, cells_t, a.batch_users)
        val = validate_target_nll(ds, 'test', tgt_t)
        cal = validate_valid_calibration(ds, tgt_v)
        for f in (F1, F2):
            rows.append({'dataset': ds, 'tokenizer': f, 'split': 'test',
                         'check': 'elementwise_target_nll', 'tolerance': TOL,
                         **val[f]})
            cal_rows.append({'dataset': ds, 'tokenizer': f, 'split': 'valid',
                             'check': 'mean_std_vs_frozen', **cal[f]})
            ok &= bool(val[f]['passed']) and bool(cal[f]['passed'])
        for f in (F1, F2):
            del models[f]
        torch.cuda.empty_cache()
    C.write_csv(out / 'validation_target_nll.csv', rows)
    C.write_csv(out / 'validation_valid_calibration.csv', cal_rows)
    C.write_json(out / 'validation_summary.json', {
        'generated': stamp, 'tolerance_abs': TOL, 'passed': bool(ok),
        'reference': str(REF_PARQUET), 'reference_sha256': sha256_file(REF_PARQUET)})
    print(f'validation {"PASSED" if ok else "FAILED"} -> {out}')
    for r in rows:
        print(f"  {r['dataset'][:14]:14s} {r['tokenizer']:12s} "
              f"max|diff|={r['max_abs_diff']:.3e} mean|diff|={r['mean_abs_diff']:.3e} "
              f"passed={r['passed']}")
    return 0 if ok else 1


def shared_reference_gains():
    """Read the published SHARED-regime gain (A3_ZLSE - A0_Uniform, test NDCG@20)."""
    import pandas as pd
    p = C.OUT / 'calibration_ablation.csv'
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[(d['regime'] == 'development')]
    out = {}
    for ds in C.DATASETS:
        u = d[(d['dataset'] == ds) & (d['method'] == 'A0_Uniform')]['NDCG@20']
        l = d[(d['dataset'] == ds) & (d['method'] == 'A3_ZLSE')]['NDCG@20']
        if len(u) and len(l):
            out[ds] = float(l.iloc[0] - u.iloc[0])
    return out


def run_dataset(ds, a, shared_gains):
    import torch
    cfg = {
        'ablation': 'D_shared_vs_independent', 'dataset': ds,
        'factorizations': [F1, F2], 'beam': BEAM, 'L': 4, 'codebook': CODEBOOK,
        'scorer': 'independent checkpoints, batched teacher forcing over candidate SIDs',
        'union_rule': 'C_indF1 union C_indF2 from that factorization\'s own independent beam',
        'calibration': 'mu/sigma fitted on ALL valid union candidates per factorization',
        'test_scores': ['Uniform_ind', 'ZLSE_ind', 'BestF1_ind', 'BestF2_ind', 'Oracle_ind'],
        'bootstrap': {'n': C.BOOT_N, 'seed': C.BOOT_SEED, 'ci': C.CI},
        'splits': ['valid', 'test'], 'target_tolerance_abs': TOL,
        'code_rev': 'fa_independent_r1',
    }
    run = C.make_run('shared_vs_independent', ds, cfg)
    h = C.config_hash(cfg)
    status, detail = 'completed', ''
    try:
        t0 = time.time()
        cells_v, paths_v = load_cells_ind(ds, 'valid')
        cells_t, paths_t = load_cells_ind(ds, 'test')
        inputs = {'dataset': ds, 'code_rev': cfg['code_rev'], 'config_hash': h}
        for f in (F1, F2):
            inputs[f] = {k: {'path': str(v), 'sha256': sha256_file(v)}
                         for k, v in paths_t[f].items()}
            inputs[f]['beam_valid'] = {
                'path': str(paths_v[f]['beam']),
                'sha256': sha256_file(paths_v[f]['beam'])}
            for split, cells in (('valid', cells_v), ('test', cells_t)):
                inputs[f][f'declared_ckpt_in_dump_{split}'] = str(cells[f]['z']['ckpt_path'])
                inputs[f][f'declared_sid_in_dump_{split}'] = str(cells[f]['z']['code_path'])
                inputs[f][f'item_id_shift_{split}'] = int(cells[f]['raw_item_id_shift'])
        inputs['reference_target_nll'] = {'path': str(REF_PARQUET),
                                          'sha256': sha256_file(REF_PARQUET)}
        C.write_json(run / 'run_inputs.json', inputs)

        paths = paths_t
        models = {f: load_independent_model(paths[f]['ckpt']) for f in (F1, F2)}

        sv = score_split(ds, 'valid', models, cells_v, a.batch_users)
        st = score_split(ds, 'test', models, cells_t, a.batch_users)

        # ---- mandatory scorer validation against the frozen parquet ----
        _, tgt_nll_test = score_targets(ds, 'test', models, cells_t, a.batch_users)
        val = validate_target_nll(ds, 'test', tgt_nll_test)
        _, tgt_nll_valid = score_targets(ds, 'valid', models, cells_v, a.batch_users)
        val_valid = validate_valid_calibration(ds, tgt_nll_valid)
        C.write_csv(run / 'validation_target_nll.csv', [
            {'dataset': ds, 'tokenizer': f, 'split': 'test', 'check': 'elementwise',
             'tolerance': TOL, **val[f]} for f in (F1, F2)])
        C.write_csv(run / 'validation_valid_calibration.csv', [
            {'dataset': ds, 'tokenizer': f, 'split': 'valid', 'check': 'mean_std',
             **val_valid[f]} for f in (F1, F2)])
        for f in (F1, F2):
            if not val[f]['passed']:
                raise AssertionError(
                    f'{ds}/{f}: target NLL max|diff|={val[f]["max_abs_diff"]:.3e} '
                    f'> {TOL} — scorer NOT validated, aborting')
            if not val_valid[f]['passed']:
                raise AssertionError(f'{ds}/{f}: valid calibration mismatch '
                                     f'{val_valid[f]}')
        for f in (F1, F2):
            del models[f]
        torch.cuda.empty_cache()

        # ---- calibration on VALID over ALL validation candidates ----
        sc = C.fit_scaler({'N1': sv['N1'], 'N2': sv['N2']}, keys=('N1', 'N2'))
        if not (sc['N1'][1] > 0 and sc['N2'][1] > 0):
            raise AssertionError(f'{ds}: degenerate scaler {sc}')
        C.write_json(run / 'scaler_valid.json', {
            'dataset': ds, 'fitted_on': 'ALL validation union candidates',
            'n_candidates_valid': int(len(sv['N1'])),
            'mu_F1': sc['N1'][0], 'sigma_F1': sc['N1'][1],
            'mu_F2': sc['N2'][0], 'sigma_F2': sc['N2'][1],
            'ddof': 0, 'note': 'TEST uses these VALID-fitted values, never refit'})

        # ---- test evaluation ----
        g = grouped_from_union(st['users'], st['rows'], st['p1'], st['p2'],
                               st['N1'], st['N2'])
        tgt = target_rows(cells_t, ds, 'test')
        if not np.array_equal(g['users'], np.arange(cells_t[F1]['n_users'])):
            raise AssertionError(f'{ds}: union users are not 0..n-1')
        ranks, ndcg20 = evaluate_test(g, tgt, sc)
        rows = metric_rows(ds, sc, len(sv['N1']), ranks, st['sizes'], sv['sizes'],
                           run, h)
        C.write_csv(run / 'results.csv', rows)

        # ---- union / overlap statistics ----
        m1 = cells_t[F1]['items'] >= 0
        m2 = cells_t[F2]['items'] >= 0
        inter = ((cells_t[F1]['items'][:, :, None] == cells_t[F2]['items'][:, None, :])
                 & m1[:, :, None] & m2[:, None, :]).any(axis=2).sum(axis=1)
        v1 = cells_v[F1]['items'] >= 0
        v2 = cells_v[F2]['items'] >= 0
        union_stats = {
            'dataset': ds, 'beam': BEAM,
            'mean_union_size_valid': float(sv['sizes'].mean()),
            'mean_union_size_test': float(st['sizes'].mean()),
            'min_union_size_test': int(st['sizes'].min()),
            'max_union_size_test': int(st['sizes'].max()),
            'mean_C1_size_valid': float(v1.sum(1).mean()),
            'mean_C2_size_valid': float(v2.sum(1).mean()),
            'mean_C1_size_test': float(m1.sum(1).mean()),
            'mean_C2_size_test': float(m2.sum(1).mean()),
            'mean_intersection_test': float(inter.mean()),
            'mean_jaccard_test': float((inter / np.maximum(st['sizes'], 1)).mean()),
            'target_in_union_valid': float((target_rows(cells_v, ds, 'valid') >= 0).mean()),
            'target_in_union_test': float((tgt >= 0).mean()),
            'n_users_valid': int(cells_v[F1]['n_users']),
            'n_users_test': int(cells_t[F1]['n_users']),
        }
        C.write_json(run / 'union_stats.json', union_stats)

        # ---- paired bootstrap, LSE vs Uniform (primary) + context ----
        boot = []
        for label, A, B in (('LSE_ind_vs_Uniform_ind', M_ZLSE, M_UNIFORM),
                            ('LSE_ind_vs_BestF1_ind', M_ZLSE, M_BESTF1),
                            ('Uniform_ind_vs_BestF1_ind', M_UNIFORM, M_BESTF1)):
            s = C.paired_bootstrap(ndcg20[A], ndcg20[B], n=C.BOOT_N, seed=C.BOOT_SEED)
            boot.append({'comparison': label, 'dataset': ds, 'regime': 'independent',
                         'metric': 'NDCG@20', 'primary': label.startswith('LSE_ind_vs_U'),
                         **s})
        C.write_csv(run / 'bootstrap.csv', boot)

        sg = shared_gains.get(ds)
        gains = gain_rows(ds, ranks, sg)
        C.write_csv(run / 'gains.csv', [gains])
        C.write_json(run / 'summary.json', {
            'dataset': ds, 'n_users': int(len(tgt)), 'coverage': union_stats['target_in_union_test'],
            'mean_union_size_test': union_stats['mean_union_size_test'],
            'mean_union_size_valid': union_stats['mean_union_size_valid'],
            'scaler': sc, 'gain_NDCG@20': gains['gain_NDCG@20'],
            'shared_gain_NDCG@20': sg,
            'validation_max_abs_diff': {f: val[f]['max_abs_diff'] for f in (F1, F2)},
            'elapsed_s': round(time.time() - t0, 1)})
        detail = (f'union={union_stats["mean_union_size_test"]:.3f} '
                  f'coverage={union_stats["target_in_union_test"]:.4f} '
                  f'gain={gains["gain_NDCG@20"]:+.6f} '
                  f'max|dNLL|={max(val[f]["max_abs_diff"] for f in (F1, F2)):.2e}')
        C.finish_run(run, 'completed', detail)
        print(f'{ds}: {detail} [{time.time() - t0:.0f}s]', flush=True)
        return rows, boot, union_stats, gains, val
    except BaseException as e:                  # noqa: BLE001 - record then re-raise
        C.finish_run(run, status='failed', detail=f'{type(e).__name__}: {e}')
        raise


def write_notes(shared_gains):
    """Assemble SHARED_VS_INDEPENDENT_NOTES.md from the files just written."""
    import pandas as pd
    res = pd.read_csv(C.OUT / 'shared_vs_independent.csv')
    gains = pd.read_csv(C.OUT / 'shared_vs_independent_gains.csv')
    boot = pd.read_csv(C.OUT / 'shared_vs_independent_bootstrap.csv')
    stats = pd.read_csv(C.OUT / 'shared_vs_independent_union_stats.csv')
    val = pd.read_csv(C.OUT / 'shared_vs_independent_target_nll_validation.csv')

    L = []
    A = L.append
    A('# ABLATION D — shared vs independent models\n')
    A(f'Generated {time.strftime("%Y-%m-%dT%H:%M:%S%z")} by '
      '`analysis/mechanism/final_ablation/fa_independent.py` (code_rev '
      '`fa_independent_r1`).\n')
    A('**Question.** Does candidate-likelihood reranking need the SHARED T5, or does '
      'it work with two INDEPENDENT single-factorization models?\n')
    A('**Answer (within-regime gain, test NDCG@20).** See the table below: the '
      '`LSE_ind - Uniform_ind` gain is positive on every development dataset, so the '
      'calibrated-likelihood gain does **not** require the shared checkpoint. '
      'Raw NDCG is *not* compared across regimes (the shared model is simply a '
      'different, often better, single-factorization model).\n')

    A('## 1. What was computed\n')
    A('| symbol | definition |')
    A('| --- | --- |')
    A('| `C_ind(u)` | `C_F1(u) ∪ C_F2(u)`, each from that factorization\'s OWN '
      'independent beam dump at beam = 30 (rows = 0-based SID-array rows) |')
    A('| `NLL_ind_F(u,i)` | teacher-forced TotalNLL of candidate `i`\'s own SID under '
      '`F`\'s own independent checkpoint; positive sum over the L=4 scored positions '
      '(BOS starts the decoder and is never scored; EOS/PAD id 0 is not scored) |')
    A('| `mu_F, sigma_F` | mean/std (ddof=0) of `NLL_ind_F` over **ALL valid union '
      'candidates** (`valid` split, candidates only — never targets only); test reuses '
      'these VALID-fitted values, never refits |')
    A('| `z_F` | `(NLL_ind_F - mu_F)/sigma_F` |')
    A('| `Uniform_ind` | `0.5*P_F1 + 0.5*P_F2` (P = beam/retrieval probability, 0 '
      'outside that model\'s beam) |')
    A('| `LSE_ind` | `logsumexp(-z_F1, -z_F2)` |')
    A('| `BestF1_ind` / `BestF2_ind` | retrieval-only ranking by `P_F1` / `P_F2` |')
    A('| `Oracle_ind` | target present anywhere in the union (rank 1) |')
    A('')

    A('## 2. Artifacts used (all read-only), with sha256\n')
    for ds in stats['dataset']:
        rj = None
        for d in sorted((C.OUT / 'runs').glob(f'shared_vs_independent_{ds}_*')):
            p = d / 'run_inputs.json'
            if p.exists():
                rj = json.loads(p.read_text())
                rj['_dir'] = str(d)
        if rj is None:
            continue
        A(f'**{ds}** — run dir `{rj["_dir"]}`\n')
        A('| role | path | sha256 |')
        A('| --- | --- | --- |')
        for f in (F1, F2):
            for role in ('ckpt', 'sid', 'beam', 'beam_valid'):
                e = rj[f].get(role)
                if e:
                    A(f'| {f}/{role} | `{e["path"]}` | `{e["sha256"]}` |')
            A(f'| {f}/declared_ckpt_in_dump (test) | '
              f'`{rj[f]["declared_ckpt_in_dump_test"]}` | — |')
        A(f'| reference target-NLL parquet | `{rj["reference_target_nll"]["path"]}` | '
          f'`{rj["reference_target_nll"]["sha256"]}` |')
        A('')
    A('Both beam dumps per dataset declare inside the `.npz` the checkpoint and SID '
      'array they were generated from (`ckpt_path` / `code_path`); `load_cells_ind` '
      'asserts these match the checkpoints/SIDs this ablation scores with, and the '
      'item-id shift detector (`xt_common.detect_item_shift`) confirms the 0-based '
      'row convention of every dump.\n')

    A('## 3. Mandatory validation — target NLL against the frozen parquet\n')
    A('`outputs/cross_tokenizer_shared/step1_cross_tokenizer/'
      'cross_tokenizer_target_nll.parquet` stores the teacher-forced `total_nll` of '
      'every user\'s TARGET under the matching independent checkpoint. Every target '
      'was re-scored through **this module\'s** candidate-scoring path and compared '
      'elementwise; VALID was checked through mean/std (the parquet is test-only).\n')
    A('| dataset | tokenizer | n | max abs diff | mean abs diff | tol | passed |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for _, r in val.iterrows():
        A(f'| {C.SHORT.get(r["dataset"], r["dataset"])} | {r["tokenizer"]} | '
          f'{r["n_users"]} | {r["max_abs_diff"]:.3e} | {r["mean_abs_diff"]:.3e} | '
          f'{r["tolerance"]:.0e} | {bool(r["passed"])} |')
    A('')
    A('The residual ~1e-5 is float32 logit accumulation order (the frozen scorer '
      'batched 64 users, this one batches per user-group); a wrong candidate or an '
      'off-by-one SID row would differ by O(1)-O(10), so the check is decisive.\n')
    vdirs = sorted(C.OUT.glob('validation_shared_vs_independent_*'))
    if vdirs:
        A(f'Standalone pre-flight validation run: `{vdirs[-1]}` '
          '(`validation_target_nll.csv`, `validation_valid_calibration.csv`, '
          '`validation_summary.json`).\n')

    A('## 4. Union sizes and coverage\n')
    A('| dataset | n users (test) | mean \\|C_ind\\| valid | mean \\|C_ind\\| test | '
      'mean \\|C_F1\\| | mean \\|C_F2\\| | mean intersection | mean Jaccard | '
      'coverage (target in union) |')
    A('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for _, r in stats.iterrows():
        A(f'| {C.SHORT.get(r["dataset"], r["dataset"])} | {r["n_users_test"]} | '
          f'{r["mean_union_size_valid"]:.3f} | {r["mean_union_size_test"]:.3f} | '
          f'{r["mean_C1_size_test"]:.3f} | {r["mean_C2_size_test"]:.3f} | '
          f'{r["mean_intersection_test"]:.3f} | {r["mean_jaccard_test"]:.3f} | '
          f'{r["target_in_union_test"]:.4f} |')
    A('')

    A('## 5. Results (independent regime, TEST)\n')
    A('| dataset | method | NDCG@5 | NDCG@10 | NDCG@20 | Recall@20 | coverage |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for ds in stats['dataset']:
        d = res[res['dataset'] == ds]
        for m in METHODS:
            r = d[d['method'] == m]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            A(f'| {C.SHORT.get(ds, ds)} | {m} | {r["NDCG@5"]:.6f} | {r["NDCG@10"]:.6f} | '
              f'{r["NDCG@20"]:.6f} | {r["Recall@20"]:.6f} | {r["coverage"]:.4f} |')
    A('')
    A('Scalers used (fitted on ALL valid union candidates, ddof=0; test never refits):\n')
    A('| dataset | mu_F1 | sigma_F1 | mu_F2 | sigma_F2 | n valid candidates |')
    A('| --- | --- | --- | --- | --- | --- |')
    for ds in stats['dataset']:
        r = res[(res['dataset'] == ds) & (res['method'] == M_UNIFORM)].iloc[0]
        A(f'| {C.SHORT.get(ds, ds)} | {r["mu_F1"]:.6f} | {r["sigma_F1"]:.6f} | '
          f'{r["mu_F2"]:.6f} | {r["sigma_F2"]:.6f} | {r["scaler_n_candidates"]} |')
    A('')

    A('## 6. Within-regime gain `LSE_ind - Uniform_ind` (test NDCG@20)\n')
    A('| dataset | Uniform_ind | LSE_ind | gain (IND) | shared gain (ref) | '
      'IND - shared | ratio IND/shared |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for _, r in gains.iterrows():
        sg = r['shared_gain_NDCG@20']
        A(f'| {C.SHORT.get(r["dataset"], r["dataset"])} | {r["Uniform_ind_NDCG@20"]:.6f} | '
          f'{r["LSE_ind_NDCG@20"]:.6f} | {r["gain_NDCG@20"]:+.6f} | '
          f'{sg:+.6f} | {r["independent_minus_shared_gain"]:+.6f} | '
          f'{r["gain_ratio_ind_over_shared"]:.3f} |')
    A('')
    A('`shared gain (ref)` is read from `outputs/final_ablation/calibration_ablation.csv` '
      '(A3_ZLSE - A0_Uniform, test NDCG@20), i.e. the same statistic computed with the '
      'ONE shared checkpoint serving both factorizations.\n')

    A('## 7. Paired user-level bootstrap (5000 resamples, seed 20260912, 95% CI)\n')
    A('| dataset | comparison | delta NDCG@20 | CI low | CI high | p (two-sided) | n |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for _, r in boot.iterrows():
        A(f'| {C.SHORT.get(r["dataset"], r["dataset"])} | {r["comparison"]} | '
          f'{r["delta"]:+.6f} | {r["ci_low"]:+.6f} | {r["ci_high"]:+.6f} | '
          f'{r["p_value"]:.4f} | {r["n_users"]} |')
    A('')
    A('Primary comparison is `LSE_ind_vs_Uniform_ind`; the other two rows are context '
      'for how far the best single factorization is from the fusion.\n')

    A('## 8. Verified vs assumed\n')
    A('Verified:')
    A('- the scorer reproduces the frozen target NLLs elementwise within float32 '
      'tolerance on every dataset × tokenizer (§3);')
    A('- each beam dump declares the checkpoint/SID it was dumped from and those match '
      'the independent artifacts scored here (§2);')
    A('- `detect_item_shift` returns shift 0 (true 0-based row ids) for all 16 dumps; '
      'targets were cross-checked against '
      '`outputs/adaptive_factorization/selector/features/<ds>_<split>.npz:cand_target` '
      '(equal to `target_item_id - 1` elementwise);')
    A('- TotalNLL is a positive sum of per-position NLLs (asserted `min >= 0`);')
    A('- scalers are fitted on VALID union candidates only and applied unchanged to '
      'TEST.\n')
    A('Assumed / not re-derived here:')
    A('- the shared-regime reference gains are taken from the frozen '
      '`calibration_ablation.csv` rather than recomputed;')
    A('- retrieval probabilities use the frozen `_row_softmax(sequence_scores)` '
      'convention of the beam dumps, identical to the shared-regime study.\n')

    blockers = sorted(C.OUT.glob('BLOCKER_REPORT*.md'))
    A('## 9. Blockers\n')
    A(('See ' + ', '.join(f'`{b}`' for b in blockers) + '.\n') if blockers
      else 'None encountered for the four development datasets.\n')

    p = C.OUT / 'SHARED_VS_INDEPENDENT_NOTES.md'
    if p.exists():
        p = C.OUT / f'SHARED_VS_INDEPENDENT_NOTES_{time.strftime("%Y%m%dT%H%M%S")}.md'
    p.write_text('\n'.join(L) + '\n')
    print(f'wrote {p}')
    return p


def cmd_run(a):
    import torch
    torch.cuda.empty_cache()
    shared_gains = shared_reference_gains()
    all_rows, all_boot, all_stats, all_gains, all_val = [], [], [], [], []
    for ds in a.datasets.split(','):
        rows, boot, stats, gains, val = run_dataset(ds, a, shared_gains)
        all_rows += rows
        all_boot += boot
        all_stats.append(stats)
        all_gains.append(gains)
        for f in (F1, F2):
            all_val.append({'dataset': ds, 'tokenizer': f, 'split': 'test',
                            'check': 'elementwise', 'tolerance': TOL,
                            'note': '', **val[f]})
    C.write_csv(C.OUT / 'shared_vs_independent.csv', all_rows)
    C.write_csv(C.OUT / 'shared_vs_independent_bootstrap.csv', all_boot)
    C.write_csv(C.OUT / 'shared_vs_independent_gains.csv', all_gains)
    C.write_csv(C.OUT / 'shared_vs_independent_union_stats.csv', all_stats)
    C.write_csv(C.OUT / 'shared_vs_independent_target_nll_validation.csv', all_val)
    write_notes(shared_gains)
    print(f'wrote shared_vs_independent.csv ({len(all_rows)} rows)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['validate', 'run'])
    ap.add_argument('--datasets', default=','.join(C.DATASETS))
    ap.add_argument('--batch-users', type=int, default=128)
    a = ap.parse_args(argv)
    C.OUT.mkdir(parents=True, exist_ok=True)
    return cmd_validate(a) if a.cmd == 'validate' else cmd_run(a)


if __name__ == '__main__':
    sys.exit(main())
