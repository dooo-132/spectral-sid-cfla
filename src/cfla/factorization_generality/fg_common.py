#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared primitives for the factorization-generality study.

Scope
-----
Read-only over the frozen audited repository (`CFLA_ARTIFACT_ROOT`).  Writes
nothing.  All new outputs go under
`outputs/mechanism/factorization_generality/`.

The one structural idea here is that a "model" is any (dataset, family, K, seed)
artifact triple, and every retrieval quantity -- single-model rank, uniform
fusion over an arbitrary subset, oracle over an arbitrary subset, top-N sets --
is computed by ONE vectorised code path so that G-, P- and S-diversity pairs
cannot be compared through different machinery.

Item-id convention
------------------
Historical beam dumps store 0-based item ids (see docs/ITEM_ID_CONVENTION.md);
`canonicalize_item_ids` shifts them to 1-based.  This module keeps every id
1-based after load.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analysis.per_user import (  # noqa: E402
    DATASETS, K_LIST, sha256_file, paired_bootstrap, per_user_ndcg_vector,
)
from src.fusion.common import _row_softmax, canonicalize_item_ids  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402

OUT = REPO / 'outputs' / 'mechanism' / 'factorization_generality'
FIGDIR = OUT / 'figures'

# family label -> artifact suffix used in file names
FAMILIES = {
    'spectral': 'nonneg_pearson_v3',
    'even': 'evensplit_v3',
    'random': 'randomsplit_v3',
}
FAMILY_LABELS = list(FAMILIES)
CODEBOOK = 256
BEAM = 30
SEED_CANONICAL = 2025

NBIG = 10 ** 9  # sentinel key for invalid candidates when grouping


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def sid_array_path(dataset, family, k):
    d = rp.ARTIFACT_ROOT / 'data' / dataset
    suffix = FAMILIES[family]
    if family == 'spectral':
        return d / f'{dataset}_t5_spectral_{suffix}_k{k}_sid{k + 1}.npy'
    return d / f'{dataset}_t5_{suffix}_k{k}_sid{k + 1}.npy'


def ckpt_path(dataset, family, k):
    return (rp.ARTIFACT_ROOT / 'model' / 'ckpt' /
            f'tiger_{dataset}_{FAMILIES[family]}_k{k}_sid{k + 1}.pth')


def beam_path(dataset, family, k, split='test'):
    return (rp.ARTIFACT_ROOT / f'model/beam_dumps_{dataset}_{FAMILIES[family]}' /
            f'{split}_k{k}_sid{k + 1}.npz')


SEED_ROOT = rp.ARTIFACT_ROOT / 'experiments' / 'seed_ensemble'


def seed_ckpt_path(dataset, k, seed):
    return SEED_ROOT / 'ckpt' / f'tiger_{dataset}_k{k}_sid{k + 1}_seed{seed}.pth'


def seed_beam_path(dataset, k, seed, split='test'):
    """Seed beams: prefer newly generated ones in this repo, else historical.

    The audited repository is READ-ONLY, so beams generated for the S-diversity
    control are written under `outputs/.../seed_beams/` and shadow nothing.
    """
    name = f'{dataset}_seed{seed}_{split}_k{k}_sid{k + 1}.npz'
    local = OUT / 'seed_beams' / name
    if local.exists():
        return local
    return SEED_ROOT / 'beams' / name


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
_SID_CACHE: dict = {}


def load_sid(dataset, family, k):
    key = (dataset, family, k)
    if key not in _SID_CACHE:
        _SID_CACHE[key] = np.load(sid_array_path(dataset, family, k),
                                  allow_pickle=True).astype(np.int64)
    return _SID_CACHE[key]


def detect_item_shift(z, sid, n_users=120):
    """Recover the item-id convention of ONE beam dump from its own contents.

    Returns `shift` with the meaning **row = raw_id - shift**, so the canonical
    1-based item id is `raw_id - shift + 1`.  Two conventions exist in this
    project and they must never be mixed:

      shift=0  `candidate_item_ids` is the 0-based SID-array row index
               (every historical family dump under `model/beam_dumps_*`);
      shift=1  it is already the 1-based canonical item id
               (the `experiments/seed_ensemble/beams/` dumps, produced by the
               later "fixed" dumper).

    The discriminator is exact: the dumped tokens must equal
    `sid[raw - shift] + position*256 + 1`.
    """
    tok = np.asarray(z['candidate_tokens'])
    raw = np.asarray(z['candidate_item_ids'])
    L = tok.shape[2]
    pos = np.arange(L, dtype=np.int64)[None, :]
    n = min(n_users, raw.shape[0])
    scores = {}
    for shift in (0, 1):
        idx = raw[:n] - shift
        ok = tot = 0
        for i in range(n):
            m = idx[i] >= 0
            if not m.any():
                continue
            ids = idx[i][m]
            if ids.max() >= len(sid):
                continue
            rec = sid[ids] + pos * CODEBOOK + 1
            ok += int((rec == tok[i][m]).all(axis=1).sum())
            tot += int(m.sum())
        scores[shift] = (ok / tot) if tot else float('nan')
    best = max(scores, key=lambda s: scores[s] if np.isfinite(scores[s]) else -1.0)
    if not np.isfinite(scores[best]) or scores[best] < 0.99:
        raise AssertionError(f'cannot determine item-id shift: {scores}')
    return int(best), scores


def load_cell(dataset, family, k, split='test', seed=SEED_CANONICAL,
              beam_override=None, keep_z=False, want_sha=False):
    """Load one model's beam dump.

    Returns a dict with canonical 1-based item ids, row-softmax beam
    probabilities, canonical targets, the detected raw item-id shift, and
    (optionally) the raw tokens.
    """
    if beam_override is not None:
        p = Path(beam_override)
    elif seed == SEED_CANONICAL:
        p = beam_path(dataset, family, k, split)
    else:
        p = seed_beam_path(dataset, k, seed, split)
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    raw_items = np.asarray(z['candidate_item_ids'])
    raw_target = np.asarray(z['target_item_ids'])
    sid = load_sid(dataset, family, k)
    shift, shift_scores = detect_item_shift(z, sid)
    # `shift` satisfies row = raw - shift; the canonical item id is row + 1.
    # Historical dumps: raw = row (shift 0).  1-based dumps: raw = row + 1
    # (shift 1).
    items = np.where(raw_items >= 0, raw_items - shift + 1, -1).astype(np.int64)
    target = np.where(raw_target >= 0, raw_target - shift + 1, -1).astype(np.int64)
    if target.max() > len(sid):
        raise AssertionError(f'{p}: target {target.max()} exceeds {len(sid)} items')
    scores = np.asarray(z['sequence_scores'], dtype=np.float64)
    probs = _row_softmax(scores, raw_items >= 0)
    if not np.isfinite(probs).all():
        raise AssertionError(f'non-finite probabilities in {p}')
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError(f'beam probabilities do not sum to 1 in {p}')
    cell = {
        'dataset': dataset, 'family': family, 'k': k, 'seed': seed,
        'split': split, 'path': p,
        'raw_item_id_shift': shift, 'shift_detection': shift_scores,
        'items': items,                       # (n, B) 1-based, -1 invalid
        'probs': probs,                       # (n, B) sums to 1 per user
        'target': target,                     # (n,)
        'n_users': int(items.shape[0]), 'beam': int(items.shape[1]),
    }
    if keep_z:
        cell['z'] = z
        cell['tokens'] = np.asarray(z['candidate_tokens'])
    if want_sha:
        cell['sha256'] = sha256_file(p)
    return cell


def model_name(family, k, seed=SEED_CANONICAL):
    if seed == SEED_CANONICAL:
        return f'{family}_k{k}'
    return f'{family}_k{k}_seed{seed}'


# ---------------------------------------------------------------------------
# vectorised union fusion over an arbitrary model list
# ---------------------------------------------------------------------------
def _grouped_probabilities(items2d, probs2d):
    """Merge columns that decode to the SAME item, per row.

    Returns (keys, score, real):
      keys  (n, C) sorted item ids, `NBIG` in invalid slots
      score (n, C) sum of probabilities of that item, -inf outside the first
            column of each item group
      real  (n, C) True at the first column of each valid item group

    Implementation note: the accumulation buffer is indexed by GROUP id, so it
    must be gathered back through `grp` before it can be compared column-wise
    with `keys`; reading it directly would compare each item against another
    item's probability.
    """
    valid = items2d >= 0
    key = np.where(valid, items2d, NBIG)
    order = np.argsort(key, axis=1, kind='stable')
    ks = np.take_along_axis(key, order, axis=1)
    vs = np.take_along_axis(probs2d, order, axis=1)

    first = np.empty(ks.shape, dtype=bool)
    first[:, 0] = True
    first[:, 1:] = ks[:, 1:] != ks[:, :-1]
    grp = np.cumsum(first, axis=1) - 1

    acc = np.zeros_like(vs)
    rows = np.repeat(np.arange(ks.shape[0], dtype=np.int64), ks.shape[1])
    np.add.at(acc, (rows, grp.reshape(-1)), vs.reshape(-1))
    grouped = np.take_along_axis(acc, grp, axis=1)

    real = first & (ks < NBIG)
    return ks, np.where(real, grouped, -np.inf), real


def union_fuse(cells, weights=None):
    """Uniform (or weighted) item-level probability fusion over `cells`.

    P_S(i|u) = sum_m w_m * P_m(i|u), on the union of the cells' candidates.

    Returns dict with per-user:
      rank           1-based fused rank of the target (-1 if absent)
      target_score   fused probability of the target (-inf if absent)
      top_items      (n, TOPN) item ids of the fused top-TOPN (-1 padding)
      top_scores     (n, TOPN) corresponding fused probabilities (-inf padding)
      n_valid_top    (n,) number of valid entries in top_items
      union_size     (n,) size of the fused candidate union
    """
    n = cells[0]['items'].shape[0]
    B = cells[0]['items'].shape[1]
    M = len(cells)
    for c in cells:
        if c['items'].shape != (n, B):
            raise AssertionError('cells disagree on (n, beam)')
        if not np.array_equal(c['target'], cells[0]['target']):
            raise AssertionError('cells disagree on targets')

    items = np.stack([c['items'] for c in cells], axis=1).reshape(n, M * B)
    probs = np.stack([c['probs'] for c in cells], axis=1).reshape(n, M * B)
    if weights is None:
        w = np.full(M, 1.0 / M)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    probs = probs * np.repeat(w, B)[None, :]

    ks, score, real = _grouped_probabilities(items, probs)

    target = cells[0]['target']
    tgt_match = (ks == target[:, None]) & real
    tgt_score = np.where(tgt_match, score, -np.inf).max(axis=1)
    present = np.isfinite(tgt_score)
    rank = np.full(n, -1, dtype=np.int64)
    if present.any():
        rank[present] = 1 + (score[present] > tgt_score[present][:, None]).sum(axis=1)

    union_size = real.sum(axis=1)

    TOPN = min(20, ks.shape[1])
    part = np.argpartition(-score, TOPN - 1, axis=1)[:, :TOPN]
    top_scores = np.take_along_axis(score, part, axis=1)
    top_items = np.take_along_axis(ks, part, axis=1)
    order2 = np.argsort(-top_scores, axis=1, kind='stable')
    top_scores = np.take_along_axis(top_scores, order2, axis=1)
    top_items = np.take_along_axis(top_items, order2, axis=1)
    good = np.isfinite(top_scores)
    top_items = np.where(good, top_items, -1)
    top_scores = np.where(good, top_scores, -np.inf)
    n_valid_top = good.sum(axis=1)

    return {
        'rank': rank, 'target_score': tgt_score, 'target_present': present,
        'top_items': top_items, 'top_scores': top_scores,
        'n_valid_top': n_valid_top, 'union_size': union_size,
        'target': target,
    }


def oracle_rank(cells):
    """min over cells of the single-model target rank; -1 if absent everywhere."""
    ranks = np.stack([single_rank(c) for c in cells], axis=0)  # (M, n)
    out = np.full(ranks.shape[1], -1, dtype=np.int64)
    pos = np.where(ranks >= 1, ranks, 10 ** 9)
    best = pos.min(axis=0)
    ok = best < 10 ** 9
    out[ok] = best[ok]
    return out


def single_rank(cell):
    """1-based rank of the target under ONE model's merged item distribution."""
    ks, score, real = _grouped_probabilities(cell['items'], cell['probs'])
    tgt = cell['target']
    tgt_match = (ks == tgt[:, None]) & real
    tgt_score = np.where(tgt_match, score, -np.inf).max(axis=1)
    present = np.isfinite(tgt_score)
    rank = np.full(len(tgt), -1, dtype=np.int64)
    if present.any():
        rank[present] = 1 + (score[present] > tgt_score[present][:, None]).sum(axis=1)
    return rank


# ---------------------------------------------------------------------------
# retrieval metrics from ranks
# ---------------------------------------------------------------------------
def summarize(rank, topk=(5, 10, 20)):
    out = {}
    for nn in topk:
        hit = (rank >= 1) & (rank <= nn)
        out[f'Recall@{nn}'] = float(hit.mean())
        nd = np.zeros(len(rank))
        nd[hit] = 1.0 / np.log2(rank[hit] + 1)
        out[f'NDCG@{nn}'] = float(nd.mean())
    out['target_present'] = float((rank >= 1).mean())
    return out


def ndcg_vec(rank, nn):
    return per_user_ndcg_vector(rank, nn)


# ---------------------------------------------------------------------------
# pairwise error diversity
# ---------------------------------------------------------------------------
def _top_set_arrays(res):
    """(n, TOPN) item ids with invalid slots replaced by unique negatives."""
    t = res['top_items'].copy()
    n, T = t.shape
    pad = -1 - np.arange(T)[None, :]            # -1, -2, ... per slot
    valid = t >= 0
    return np.where(valid, t, pad), valid


def pairwise_diversity(ra, rb, topn=20):
    """Error-diversity statistics between two fused/single result dicts.

    Definitions (fixed, see report):
      hit_disagreement   P(hit_A != hit_B) at topn
      directional        P(A hit, B miss) and P(A miss, B hit)
      phi                phi coefficient of the two hit indicators
      hit_jaccard        mean |hitA & hitB| / |hitA | hitB|
      candidate_jaccard  mean Jaccard of the two top-`topn` item SETS
      union_size         mean |topA | topB|
      rank_spearman*     mean Spearman over COMMON candidate items (see below)
      rank_kendall*      mean Kendall tau-b over COMMON candidate items
    """
    ha = (ra >= 1) & (ra <= topn)
    hb = (rb >= 1) & (rb <= topn)
    n = len(ra)
    both = int((ha & hb).sum())
    only_a = int((ha & ~hb).sum())
    only_b = int((~ha & hb).sum())
    neither = int((~ha & ~hb).sum())
    union = both + only_a + only_b
    num = both * neither - only_a * only_b
    den = np.sqrt((both + only_a) * (only_b + neither) *
                  (both + only_b) * (only_a + neither))
    return {
        'n_users': n,
        'hit_a': float(ha.mean()), 'hit_b': float(hb.mean()),
        'hit_disagreement': float((only_a + only_b) / n),
        'p_a1_b0': float(only_a / n), 'p_a0_b1': float(only_b / n),
        'directional_asymmetry': float(abs(only_a - only_b) / n),
        'phi': float(num / den) if den > 0 else float('nan'),
        'hit_jaccard': float(both / union) if union else float('nan'),
    }


def candidate_overlap(set_a, valid_a, set_b, valid_b, sa, sb):
    """Candidate-set Jaccard / union from top-N set arrays."""
    ia = set_a[:, None, :]
    ib = set_b[:, :, None]
    inter = ((ia == ib) & valid_a[:, None, :] & valid_b[:, :, None]).any(axis=2).sum(axis=1)
    na = sa.astype(np.int64)
    nb = sb.astype(np.int64)
    un = na + nb - inter
    jac = np.where(un > 0, inter / np.maximum(un, 1), np.nan)
    return {
        'candidate_jaccard': float(np.nanmean(jac)),
        'candidate_intersection': float(inter.mean()),
        'candidate_union_size': float(un.mean()),
        'topn_a': float(na.mean()), 'topn_b': float(nb.mean()),
    }


def common_rank_correlation(cell_a, cell_b, max_users=2000, seed=20260912,
                            min_common=3):
    """Mean Spearman / Kendall correlation over COMMON candidate items.

    Restricted to the intersection of the two models' candidate item sets, per
    user, as required by the study protocol.  Computed on a deterministic
    subsample of users (default 2000) that have >= `min_common` common items;
    the subsample size and the number of users used are returned so the figure
    can be read with the right denominator.
    """
    from scipy.stats import spearmanr, kendalltau

    maps_a = per_user_item_probs(cell_a)
    maps_b = per_user_item_probs(cell_b)
    n = len(maps_a)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    sp, kt, sizes = [], [], []
    for i in order:
        if len(sp) >= max_users:
            break
        common = maps_a[i].keys() & maps_b[i].keys()
        if len(common) < min_common:
            continue
        common = sorted(common)
        va = np.array([maps_a[i][c] for c in common])
        vb = np.array([maps_b[i][c] for c in common])
        if np.allclose(va, vb):
            continue
        sp.append(float(spearmanr(va, vb).statistic))
        kt.append(float(kendalltau(va, vb).statistic))
        sizes.append(len(common))
    return {
        'spearman_common': float(np.mean(sp)) if sp else float('nan'),
        'spearman_common_median': float(np.median(sp)) if sp else float('nan'),
        'kendall_common': float(np.mean(kt)) if kt else float('nan'),
        'n_users_correlated': len(sp),
        'mean_common_size': float(np.mean(sizes)) if sizes else float('nan'),
        'correlation_user_subsample': max_users,
    }


def per_user_item_probs(cell):
    """List of {item: probability} with beams merged; one entry per user."""
    items, probs = cell['items'], cell['probs']
    n, B = items.shape
    out = []
    for i in range(n):
        d = {}
        for j in range(B):
            it = int(items[i, j])
            if it < 0:
                continue
            d[it] = d.get(it, 0.0) + float(probs[i, j])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# provenance helper
# ---------------------------------------------------------------------------
def file_sha(p):
    p = Path(p)
    return sha256_file(p) if p.exists() else None


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_delta_ci(a, b, z=1.959963984540054):
    """Paired mean difference with a large-sample (normal) 95% CI.

    Used per pair: a 5,000-resample bootstrap on ~40k users, repeated for every
    one of the ~200 pairs, is needlessly expensive, and the normal
    approximation is exact to the third decimal at these sample sizes.  The
    headline comparisons in `statistical_tests.csv` use resampling instead.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = len(d)
    if n == 0:
        return {'delta': float('nan'), 'se': float('nan'), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'z': float('nan'), 'n_users': 0}
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    return {'delta': delta, 'se': se, 'ci_lo': delta - z * se,
            'ci_hi': delta + z * se,
            'z': (delta / se) if se and se > 0 else float('nan'),
            'n_users': int(n),
            'significant_95': bool(se and se > 0 and abs(delta) > z * se)}


def paired_bootstrap_chunked(a, b, n_boot=5000, seed=20260912, chunk=200):
    """Paired user-level bootstrap, chunked so memory stays bounded."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = len(d)
    if np.allclose(d, 0.0):
        return {'delta': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
                'p_value': float('nan'), 'note': 'identical'}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))
        means[s:e] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {'delta': float(d.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_value': float(min(1.0, max(p, 1.0 / n_boot))), 'n_boot': n_boot}


def bootstrap_mean_ci(values, n_boot=10000, seed=20260912):
    """Bootstrap CI for the mean of a vector of PAIR-level statistics.

    This is the right level of resampling for the study's main comparison --
    'is the G-diversity distribution of fusion gains different from the
    P-diversity distribution' -- because the sampling unit is a model pair,
    not a user.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'n': 0, 'median': float('nan'), 'std': float('nan')}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(axis=1)
    return {'mean': float(v.mean()), 'median': float(np.median(v)),
            'std': float(v.std(ddof=1)) if v.size > 1 else float('nan'),
            'min': float(v.min()), 'max': float(v.max()),
            'ci_lo': float(np.percentile(m, 2.5)),
            'ci_hi': float(np.percentile(m, 97.5)), 'n': int(v.size),
            'n_boot': n_boot}


def two_sample_bootstrap_diff(a, b, n_boot=10000, seed=20260912):
    """Bootstrap CI for mean(a) - mean(b) with independent resampling."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {'diff': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'n_a': int(a.size), 'n_b': int(b.size)}
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    obs = float(a.mean() - b.mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {'diff': obs, 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_value': float(min(1.0, max(p, 1.0 / n_boot))),
            'n_a': int(a.size), 'n_b': int(b.size), 'n_boot': n_boot,
            'significant_95': bool(lo > 0 or hi < 0)}
