#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE B — reproduce the FROZEN old DynamicFusion exactly, from its own artifacts.

The frozen method (old repo, `model/fuse_beam_outputs_dynamic.py`):

    score_F(u, i) = row-softmax(sequence_scores) accumulated over beams that decode
                    to item i                                   (per factorization F)
    Conf_F(u)     = 1 - H(p_F(u)) / log(max(B_valid, 2)),
                    p_F = the SAME row-softmax distribution
    Q_F           = structure weight from `structure_weights_<ds>_<aff>.json`
    q_norm        = z-score of Q over the four factorizations
    alpha(u)      = row-softmax(gamma * q_norm + eta * Conf(u))       (per user)
    fused(u, i)   = sum_F alpha_F(u) * score_F(u, i)

gamma and eta are the HISTORICAL frozen values (gamma = 0 everywhere; eta = 2 / 5 /
0 / 2 for Beauty / Cell Phones / Health / Toys).  Nothing is re-tuned here: the
recorded test metrics in `fusion_test_dynamic_<ds>_<aff>.json` must be reproduced.

No new hyper-parameter is searched (prompt section 12); this script only re-derives
the numbers from the raw dumps and reports the maximum absolute reproduction error.

Usage
  python3 mf_dynamic_fusion.py --splits test              # reproduction audit
  python3 mf_dynamic_fusion.py --splits valid,test --dump # + per-user scores
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (DATASETS, OUT, S4, dynamic_json, dynamic_params,  # noqa: E402
                       load_beam_cell, read_json, row_softmax, sha256_file,
                       structure_json, write_json)

TOPK = (5, 10, 20)


# ---------------------------------------------------------------------------
# the frozen formulas, transcribed from the old implementation
# ---------------------------------------------------------------------------
def confidence_from_scores(scores, valid, temperature=1.0):
    """Conf = 1 - H(p) / log(max(B_valid, 2)) with p = row-softmax(scores/T)."""
    x = np.where(valid, scores / temperature, -1e30)
    mx = x.max(axis=1, keepdims=True)
    e = np.exp(x - mx) * valid
    p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    n_valid = valid.sum(axis=1)
    denom = np.log(np.maximum(n_valid, 2))
    conf = np.where(n_valid <= 0, 0.0, 1.0 - entropy / (denom + 1e-12))
    return np.clip(conf, 0.0, 1.0).astype(np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


def build_union_tensor(cand_list, score_list):
    """Union of the four beams' items; identical ids ACCUMULATE (old semantics)."""
    n = cand_list[0].shape[0]
    m = len(cand_list)
    b = cand_list[0].shape[1]
    items = np.full((n, m * b), -1, dtype=np.int64)
    tensor = np.zeros((n, m * b, m), dtype=np.float64)
    for i in range(n):
        pos_of = {}
        pos = 0
        for mi in range(m):
            for item, sc in zip(cand_list[mi][i], score_list[mi][i]):
                item = int(item)
                if item < 0:
                    continue
                if item not in pos_of:
                    pos_of[item] = pos
                    items[i, pos] = item
                    pos += 1
                tensor[i, pos_of[item], mi] += float(sc)
    return items, tensor


def dynamic_weights(q_norm, conf, gamma, eta):
    logits = gamma * q_norm[None, :] + eta * conf
    return row_softmax(logits)


def metrics_from_ranked(ranked_items, target, topk_list=TOPK):
    """Transcribed from the old evaluator: target -1 carries 0, denominator = all rows."""
    res = {}
    valid_target = target >= 0
    for k in topk_list:
        top = ranked_items[:, :k]
        hit_mat = (top == target[:, None]) & valid_target[:, None]
        hit = hit_mat.any(axis=1)
        res[f'Recall@{k}'] = float(hit[valid_target].mean()) if valid_target.sum() else 0.0
        ndcg_vals = np.zeros(len(target), dtype=np.float64)
        rows, cols = np.where(hit_mat)
        if len(rows):
            ndcg_vals[rows] = 1.0 / np.log2(cols + 2)
        res[f'NDCG@{k}'] = float(ndcg_vals[valid_target].mean()) if valid_target.sum() else 0.0
    return res


def rank_all(items, tensor, weights, topk_list=TOPK):
    """Old ranking: argpartition(maxk) then argsort on the partitioned block."""
    fused = np.sum(tensor * weights[:, None, :], axis=2)
    fused = np.where(items >= 0, fused, -1e30)
    maxk = max(topk_list)
    idx = np.argpartition(-fused, kth=maxk - 1, axis=1)[:, :maxk]
    part = np.take_along_axis(fused, idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    return (np.take_along_axis(items, top_idx, axis=1),
            np.take_along_axis(fused, top_idx, axis=1),
            fused)


def dynamic_arrays(ds, split, gamma=None, eta=None):
    """Old DynamicFusion's per-user weights and per-candidate scores.

    Returns dict with
      conf  (n, 4) confidence per factorization
      w     (n, 4) alpha weights  (softmax over the four factorizations)
      items (n, U) raw 0-based item ids of the 4-K union
      score (n, U) fused score, -inf where padded
      target(n,)  raw 0-based target id
    The fusion is the historical one verbatim: acc-probabilities per factorization,
    per-user softmax weighting of the confidence, weighted sum, no renormalisation.
    """
    sj = read_json(structure_json(ds))
    q = np.array([r['Q'] for r in sj['records']], dtype=np.float64)
    q_norm = zscore(q)
    if gamma is None or eta is None:
        g_frozen, e_frozen = dynamic_params(ds)
        gamma = g_frozen if gamma is None else gamma
        eta = e_frozen if eta is None else eta
    cand_list, score_list, conf_list = [], [], []
    for f in S4:
        cell = load_beam_cell(f, ds, split)
        cand = cell['items']
        sq = np.load(cell['path'], allow_pickle=True)['sequence_scores'].astype(np.float64)
        valid = cand >= 0
        score_list.append(row_softmax(np.where(valid, sq, -1e30)) * valid)
        conf_list.append(confidence_from_scores(sq, valid, 1.0))
        cand_list.append(cand)
    target = load_beam_cell(S4[0], ds, split)['target']
    items, tensor = build_union_tensor(cand_list, score_list)
    conf = np.stack(conf_list, axis=1)
    w = dynamic_weights(q_norm, conf, gamma, eta)
    fused = np.sum(tensor * w[:, None, :], axis=2)
    fused = np.where(items >= 0, fused, -1e30)
    return {'conf': conf, 'w': w, 'items': items, 'score': fused, 'target': target,
            'q_raw': q, 'q_norm': q_norm, 'gamma': gamma, 'eta': eta}


def reproduce_cell(ds, split, gamma=None, eta=None, want_rows=False):
    """Run the frozen DynamicFusion on one (dataset, split).  Returns metrics etc."""
    sj = read_json(structure_json(ds))
    q = np.array([r['Q'] for r in sj['records']], dtype=np.float64)
    if len(q) != len(S4):
        raise AssertionError(f'{structure_json(ds)}: {len(q)} records != {len(S4)}')
    q_norm = zscore(q)

    frozen = read_json(dynamic_json(ds, 'test')) if split == 'test' else None
    if gamma is None or eta is None:
        g_frozen, e_frozen = dynamic_params(ds)      # never re-searched
        gamma = g_frozen if gamma is None else gamma
        eta = e_frozen if eta is None else eta

    cand_list, score_list, conf_list = [], [], []
    for f in S4:
        cell = load_beam_cell(f, ds, split)
        cand = cell['items']
        sq = np.load(cell['path'], allow_pickle=True)['sequence_scores'].astype(np.float64)
        valid = cand >= 0
        score_list.append(row_softmax(np.where(valid, sq, -1e30)) * valid)
        conf_list.append(confidence_from_scores(sq, valid, 1.0))
        cand_list.append(cand)
    target = load_beam_cell(S4[0], ds, split)['target']

    items, tensor = build_union_tensor(cand_list, score_list)
    conf = np.stack(conf_list, axis=1)
    w = dynamic_weights(q_norm, conf, gamma, eta)
    ranked, _, fused = rank_all(items, tensor, w)
    metrics = metrics_from_ranked(ranked, target)

    out = {'dataset': ds, 'split': split, 'gamma': gamma, 'eta': eta,
           'metrics': metrics,
           'avg_weights': w.mean(axis=0).tolist(),
           'std_weights': w.std(axis=0).tolist(),
           'q_raw': q.tolist(), 'q_norm': q_norm.tolist(),
           'union_size_mean': float((items >= 0).sum(axis=1).mean()),
           'structure_json': str(structure_json(ds)),
           'structure_sha256': sha256_file(structure_json(ds)),
           'beam_paths': [str(load_beam_cell(f, ds, split)['path']) for f in S4],
           'beam_sha256': [load_beam_cell(f, ds, split)['sha256'] for f in S4],
           'frozen_reference': frozen}
    if want_rows:
        out['_ranked_items'] = ranked
        out['_fused'] = fused
        out['_items'] = items
        out['_weights'] = w
        out['_target'] = target
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--splits', default='test')
    ap.add_argument('--out', default=str(OUT / 'DYNAMIC_FUSION_REPRODUCTION.json'))
    a = ap.parse_args(argv)

    report = {'method': 'old_frozen_dynamic_confidence_aware_fusion',
              'formula': 'alpha_K(u) = softmax(gamma * z(Q)_K + eta * Conf_K(u)); '
                         'score = sum_K alpha_K(u) * P_K(i|u)',
              'cells': {}, 'max_abs_error_vs_frozen': 0.0}
    worst = 0.0
    for ds in a.datasets.split(','):
        for split in a.splits.split(','):
            r = reproduce_cell(ds, split)
            key = f'{ds}|{split}'
            report['cells'][key] = {k: v for k, v in r.items()
                                    if not k.startswith('_')}
            m = r['metrics']
            line = (f"{ds:32s} {split:5s} gamma={r['gamma']:<4} eta={r['eta']:<4} "
                    f"NDCG@20={m['NDCG@20']:.6f} R@20={m['Recall@20']:.6f}")
            if split == 'test' and r['frozen_reference']:
                ref = r['frozen_reference']['metrics']
                err = max(abs(m[k] - ref[k]) for k in m)
                worst = max(worst, err)
                line += f'  |vs frozen| max={err:.3e}'
            print(line, flush=True)
    report['max_abs_error_vs_frozen'] = worst
    report['reproduced'] = bool(worst < 1e-9)
    write_json(a.out, report)
    print(f'\nmax abs error vs frozen artifacts: {worst:.3e}  '
          f'(reproduced={report["reproduced"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
