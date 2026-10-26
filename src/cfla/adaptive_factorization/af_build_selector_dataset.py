#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — build the leakage-safe target-free routing dataset.

For every dataset and every split (`valid` and `test`) this module produces one row
per user containing

  * LABELS derived from the frozen shared model's target rank under each
    factorization  ->  U_m, q_m, sample weight.  A label is allowed to use the
    target; it never enters a feature.
  * FEATURES computed only from (a) the frozen encoder representation of the user
    history and (b) the two factorizations' target-free beam outputs.

The row index of every split is the parquet row index, and the `user` column is
element-wise identical across train/valid/test and across factorizations, so
validation rows and test rows are index-aligned per user.  That is what lets the
router be supervised on the earlier (validation) event and applied to the later
(test) event without any id feature.

Outputs (per dataset) -> `<outdir>/<dataset>.npz` with arrays:
    user, split, u1, u2, q1, q2, weight, rank1, rank2,
    router_h, router_p, feature_names_h, feature_names_p

Run:
  python3 af_build_selector_dataset.py --features <features dir> --outdir <selector dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared'),
           str(HERE.parent / 'factorization_generality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xt_common import ART, CODEBOOK, DATASETS, SHORT, TOKENIZERS, load_sid, sha256_file
from fg_common import _grouped_probabilities, _row_softmax, per_user_ndcg_vector
from af_common import merge_probabilities

FACTORIZATIONS = ('spectral_k3', 'rqvae3')
T_LABEL = 0.1          # pre-declared soft-label temperature
ZERO_HIT_WEIGHT = 0.25  # pre-declared weight for examples with U1 = U2 = 0
TOPK_EVAL = 20

# ---------------------------------------------------------------------------
# feature declarations (single source of truth for the audit document)
# ---------------------------------------------------------------------------
HIST_FEATURES = ['hist_len_mean', 'hist_len_max', 'hist_len_F1_minus_F2']

CROSS_FEATURES = ['x_top20_jaccard', 'x_top20_intersection', 'x_top20_union',
                  'x_top1_item_agreement', 'x_rank_corr_common', 'x_prob_mass_l1']


def per_factor_feature_names(f):
    return [f'{f}_{n}' for n in
            ('top1_prob', 'top2_prob', 'top1_top2_margin', 'beam_entropy',
             'mass_top5', 'mass_top10', 'mass_top20', 'beam_score_mean',
             'beam_score_std', 'n_unique_items', 'collapse_ratio', 'valid_sid_ratio')]


def feature_names():
    h = ([f'h_F1_{i}' for i in range(128)] + [f'h_F2_{i}' for i in range(128)] +
         [f'h_absdiff_{i}' for i in range(128)] + HIST_FEATURES)
    p = []
    for f in FACTORIZATIONS:
        p += per_factor_feature_names(f)
    p += CROSS_FEATURES + HIST_FEATURES
    return h, p


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _grouped(items, probs):
    """Merged per-item probabilities, sorted by nothing: returns (keys, score, real)."""
    return _grouped_probabilities(items, probs)


def load_beams(feat_dir, ds, f, split):
    p = Path(feat_dir) / 'beams' / f'{ds}_{f}_{split}.npz'
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    items = np.asarray(z['candidate_item_ids']).astype(np.int64)
    target = np.asarray(z['target_item_ids']).astype(np.int64)
    probs = _row_softmax(np.asarray(z['sequence_scores'], dtype=np.float64), items >= 0)
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError(f'{p}: beam probabilities do not sum to 1')
    return {'items': items, 'probs': probs, 'target': target, 'path': p}


def load_history_repr(feat_dir, ds, f, split):
    p = Path(feat_dir) / 'history_repr' / f'{ds}_{f}_{split}.npz'
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    return np.asarray(z['history_repr'], dtype=np.float64), p


def load_history_lengths(ds, split, max_len=20):
    """Non-pad history length per row, from the parquet directly (no target used)."""
    import polars as pl
    df = pl.read_parquet(ART / 'data' / ds / f'{split}.parquet')
    lens = np.array([min(len(h), max_len) for h in df['history'].to_list()], dtype=np.float64)
    users = df['user'].to_numpy()
    return users, lens


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def target_rank(cell):
    """1-based rank of the target under one factorization's merged item distribution.

    Reuses the audited `_grouped_probabilities` machinery so ties and duplicate
    beams are handled exactly as in every prior study.
    """
    ks, score, real = _grouped(cell['items'], cell['probs'])
    tgt = cell['target']
    tgt_match = (ks == tgt[:, None]) & real
    tgt_score = np.where(tgt_match, score, -np.inf).max(axis=1)
    present = np.isfinite(tgt_score)
    rank = np.full(len(tgt), -1, dtype=np.int64)
    if present.any():
        rank[present] = 1 + (score[present] > tgt_score[present][:, None]).sum(axis=1)
    return rank


def utility_from_rank(rank, k=TOPK_EVAL):
    u = np.zeros(len(rank), dtype=np.float64)
    hit = (rank >= 1) & (rank <= k)
    u[hit] = 1.0 / np.log2(rank[hit] + 1)
    return u


def soft_targets(u1, u2, temp=T_LABEL):
    """q_m = softmax(U_m / T); (0.5, 0.5) when both utilities are 0."""
    z = np.stack([u1 / temp, u2 / temp], axis=1)          # (n, 2)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    q = e / e.sum(axis=1, keepdims=True)
    both_zero = (u1 == 0) & (u2 == 0)
    q[both_zero] = 0.5
    w = np.where(both_zero, ZERO_HIT_WEIGHT, 1.0).astype(np.float64)
    return q[:, 0], q[:, 1], w, both_zero


# ---------------------------------------------------------------------------
# target-free features
# ---------------------------------------------------------------------------
def beam_features(cell):
    """Per-factorization target-free beam statistics. Returns (n, 12) and names."""
    items, probs = cell['items'], cell['probs']
    valid = items >= 0
    n, B = items.shape
    eps = 1e-12

    order = np.argsort(-probs, axis=1, kind='stable')
    ps = np.take_along_axis(probs, order, axis=1)
    top1 = ps[:, 0]
    top2 = ps[:, 1] if B > 1 else np.zeros(n)
    p_safe = np.where(probs > 0, probs, 1.0)
    entropy = -(np.where(valid, probs * np.log(p_safe), 0.0)).sum(axis=1)
    mass5 = np.where(valid, probs, 0.0)[:, :5].sum(axis=1) if B >= 5 else probs.sum(axis=1)
    mass10 = np.where(valid, probs, 0.0)[:, :10].sum(axis=1) if B >= 10 else probs.sum(axis=1)
    mass20 = np.where(valid, probs, 0.0)[:, :20].sum(axis=1) if B >= 20 else probs.sum(axis=1)

    sc = cell['sequence_scores'] if 'sequence_scores' in cell else None
    if sc is None:
        # not stored in the cell dict; recover from the raw npz lazily elsewhere
        sc = np.zeros_like(probs)
    sc_mean = sc.mean(axis=1)
    sc_std = sc.std(axis=1)

    n_unique = np.array([len(set(items[i][valid[i]].tolist())) for i in range(n)],
                        dtype=np.float64)
    valid_sids = valid.sum(axis=1).astype(np.float64)
    collapse = np.where(n_unique > 0, valid_sids / np.maximum(n_unique, 1.0), 0.0)

    feats = np.stack([top1, top2, top1 - top2, entropy, mass5, mass10, mass20,
                      sc_mean, sc_std, n_unique, collapse, valid_sids / float(B)],
                     axis=1)
    if not np.isfinite(feats).all():
        raise AssertionError('non-finite beam feature')
    return feats


def top_items_from_cell(cell, k=TOPK_EVAL):
    """(n, k) top-k item ids by merged probability, -1 padded."""
    ks, score, real = _grouped(cell['items'], cell['probs'])
    kk = min(k, ks.shape[1])
    part = np.argpartition(-np.where(np.isfinite(score), score, -np.inf), kk - 1, axis=1)[:, :kk]
    s = np.take_along_axis(score, part, axis=1)
    it = np.take_along_axis(ks, part, axis=1)
    o = np.argsort(-s, axis=1, kind='stable')
    s = np.take_along_axis(s, o, axis=1)
    it = np.take_along_axis(it, o, axis=1)
    good = np.isfinite(s)
    return np.where(good, it, -1)


def cross_features(cell_a, cell_b, topa, topb):
    """Target-free cross-factorization agreement features. Returns (n, 6)."""
    n = topa.shape[0]
    inter = np.empty(n); union = np.empty(n); jac = np.empty(n); top1_agree = np.empty(n)
    for i in range(n):
        sa = set(int(x) for x in topa[i] if x >= 0)
        sb = set(int(x) for x in topb[i] if x >= 0)
        i_sz, u_sz = len(sa & sb), len(sa | sb)
        inter[i] = i_sz
        union[i] = u_sz
        jac[i] = (i_sz / u_sz) if u_sz else 0.0
        top1_agree[i] = 1.0 if (len(sa) and len(sb) and topa[i][0] == topb[i][0]) else 0.0

    # rank correlation over items present in BOTH factorizations' merged distributions
    ks_a, sc_a, real_a = _grouped(cell_a['items'], cell_a['probs'])
    ks_b, sc_b, real_b = _grouped(cell_b['items'], cell_b['probs'])
    rank_corr = np.zeros(n)
    for i in range(n):
        ka = ks_a[i][real_a[i]]; va = sc_a[i][real_a[i]]
        kb = ks_b[i][real_b[i]]; vb = sc_b[i][real_b[i]]
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        if common.size < 3:
            continue
        ra = 1 + (va[ia][:, None] > va[ia][None, :]).sum(axis=1)
        rb = 1 + (vb[ib][:, None] > vb[ib][None, :]).sum(axis=1)
        ra = ra.astype(np.float64); rb = rb.astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        rank_corr[i] = float((ra * rb).sum() / den) if den > 0 else 0.0

    # L1 distance between the two probability mass functions over the union of items
    mass_l1 = np.zeros(n)
    for i in range(n):
        da = {}
        for j in range(cell_a['items'].shape[1]):
            it = int(cell_a['items'][i, j])
            if it >= 0:
                da[it] = da.get(it, 0.0) + float(cell_a['probs'][i, j])
        db = {}
        for j in range(cell_b['items'].shape[1]):
            it = int(cell_b['items'][i, j])
            if it >= 0:
                db[it] = db.get(it, 0.0) + float(cell_b['probs'][i, j])
        keys = set(da) | set(db)
        mass_l1[i] = sum(abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in keys)

    return np.stack([jac, inter, union, top1_agree, rank_corr, mass_l1], axis=1)


def build_cell(feat_dir, ds, split):
    cells = {}
    scs = {}
    for f in FACTORIZATIONS:
        z = np.load(Path(feat_dir) / 'beams' / f'{ds}_{f}_{split}.npz', allow_pickle=True)
        cells[f] = load_beams(feat_dir, ds, f, split)
        cells[f]['sequence_scores'] = np.asarray(z['sequence_scores'], dtype=np.float64)
        scs[f] = cells[f]['sequence_scores']

    if not np.array_equal(cells['spectral_k3']['target'], cells['rqvae3']['target']):
        raise AssertionError(f'{ds}/{split}: the two factorizations disagree on targets')

    ranks = {f: target_rank(cells[f]) for f in FACTORIZATIONS}
    u = {f: utility_from_rank(ranks[f]) for f in FACTORIZATIONS}
    q1, q2, w, both_zero = soft_targets(u['spectral_k3'], u['rqvae3'])

    # ---- Router H ----
    h1, p1 = load_history_repr(feat_dir, ds, 'spectral_k3', split)
    h2, p2 = load_history_repr(feat_dir, ds, 'rqvae3', split)
    if h1.shape != h2.shape:
        raise AssertionError('history representations disagree in shape')
    users, lens = load_history_lengths(ds, split)
    if h1.shape[0] != len(users):
        raise AssertionError(f'{ds}/{split}: history repr rows {h1.shape[0]} != users {len(users)}')
    # History length after the canonical max_len=20 truncation is a property of the
    # parquet row, not of the factorization, so the F1/F2 difference is identically
    # zero.  It is kept as an explicit feature so that this fact is auditable rather
    # than silently assumed.
    lens = lens.astype(np.float64)
    hist_extra = np.stack([lens, lens, np.zeros_like(lens)], axis=1)
    router_h = np.concatenate([h1, h2, np.abs(h1 - h2), hist_extra], axis=1)

    # ---- compact candidate sets, so any per-user weight vector can be evaluated
    #      exactly later without regenerating beams ----
    topa = top_items_from_cell(cells['spectral_k3'])
    topb = top_items_from_cell(cells['rqvae3'])
    # k = the full beam width, so the compact set is the COMPLETE union of both
    # factorizations' candidates and `fused_rank` is exact rather than truncated.
    KCAND = cells['spectral_k3']['items'].shape[1]
    c_ids1, c_p1 = merge_probabilities(cells['spectral_k3']['items'],
                                       cells['spectral_k3']['probs'], k=KCAND)
    c_ids2, c_p2 = merge_probabilities(cells['rqvae3']['items'],
                                       cells['rqvae3']['probs'], k=KCAND)
    tgt = cells['spectral_k3']['target']

    # ---- Router P ----
    fa = beam_features(cells['spectral_k3'])
    fb = beam_features(cells['rqvae3'])
    fx = cross_features(cells['spectral_k3'], cells['rqvae3'], topa, topb)
    router_p = np.concatenate([fa, fb, fx, hist_extra], axis=1)

    names_h, names_p = feature_names()
    if router_h.shape[1] != len(names_h):
        raise AssertionError(f'router_h {router_h.shape[1]} != names {len(names_h)}')
    if router_p.shape[1] != len(names_p):
        raise AssertionError(f'router_p {router_p.shape[1]} != names {len(names_p)}')
    for name, mat in (('router_h', router_h), ('router_p', router_p)):
        if not np.isfinite(mat).all():
            raise AssertionError(f'{ds}/{split}: non-finite values in {name}')

    # ---- validity checks on the labels ----
    ndcg1 = per_user_ndcg_vector(ranks['spectral_k3'], TOPK_EVAL).mean()
    ndcg2 = per_user_ndcg_vector(ranks['rqvae3'], TOPK_EVAL).mean()

    return {
        'user': users.astype(np.int64),
        'split': np.array([split] * len(users)),
        'dataset': ds,
        'u1': u['spectral_k3'], 'u2': u['rqvae3'],
        'q1': q1, 'q2': q2, 'weight': w, 'both_zero': both_zero,
        'rank1': ranks['spectral_k3'], 'rank2': ranks['rqvae3'],
        'router_h': router_h.astype(np.float64), 'router_p': router_p.astype(np.float64),
        'feature_names_h': names_h, 'feature_names_p': names_p,
        'cand_ids1': c_ids1, 'cand_p1': c_p1.astype(np.float64),
        'cand_ids2': c_ids2, 'cand_p2': c_p2.astype(np.float64),
        'cand_target': tgt.astype(np.int64),
        'ndcg20_F1': float(ndcg1), 'ndcg20_F2': float(ndcg2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='dir holding beams/ and history_repr/')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--splits', default='valid,test')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    (outdir / 'features').mkdir(parents=True, exist_ok=True)
    prov = {'t_label': T_LABEL, 'zero_hit_weight': ZERO_HIT_WEIGHT,
            'topk_eval': TOPK_EVAL, 'features_dir': str(a.features),
            'target_in_features': False, 'cells': {}}

    for ds in a.datasets.split(','):
        for split in a.splits.split(','):
            print(f'=== build {ds} {split} ===', flush=True)
            d = build_cell(Path(a.features), ds, split)
            out_p = outdir / 'features' / f'{ds}_{split}.npz'
            np.savez_compressed(
                out_p, user=d['user'], split=d['split'],
                u1=d['u1'], u2=d['u2'], q1=d['q1'], q2=d['q2'],
                weight=d['weight'], both_zero=d['both_zero'],
                rank1=d['rank1'], rank2=d['rank2'],
                router_h=d['router_h'], router_p=d['router_p'],
                cand_ids1=d['cand_ids1'], cand_p1=d['cand_p1'],
                cand_ids2=d['cand_ids2'], cand_p2=d['cand_p2'],
                cand_target=d['cand_target'],
                feature_names_h=np.array(d['feature_names_h']),
                feature_names_p=np.array(d['feature_names_p']))
            prov['cells'][f'{ds}|{split}'] = {
                'n': int(len(d['user'])), 'file': str(out_p),
                'sha256': sha256_file(out_p),
                'ndcg20_F1_from_rank': d['ndcg20_F1'],
                'ndcg20_F2_from_rank': d['ndcg20_F2'],
                'mean_u1': float(d['u1'].mean()), 'mean_u2': float(d['u2'].mean()),
                'both_zero_fraction': float(d['both_zero'].mean()),
                'only_F1_fraction': float(((d['u1'] > 0) & (d['u2'] == 0)).mean()),
                'only_F2_fraction': float(((d['u2'] > 0) & (d['u1'] == 0)).mean()),
                'router_h_dim': int(d['router_h'].shape[1]),
                'router_p_dim': int(d['router_p'].shape[1]),
            }
            print('   ', json.dumps(prov['cells'][f'{ds}|{split}']), flush=True)

    (outdir / 'features' / 'FEATURE_BUILD_PROVENANCE.json').write_text(
        json.dumps(prov, indent=2))
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
