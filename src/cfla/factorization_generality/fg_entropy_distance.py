#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE E (structural side) + PHASE G -- identity entropy and factorization distance.

Part 1  identity_entropy_all_factorizations.csv
        For every dataset x family x K, computed exactly from the tokenizer
        mapping with no model involved:
          H(C_t | C_<t) for every position t, the marginal H(C_t), the
          cumulative semantic information curve, the semantic / disambiguation
          split, and the exact identity entropy sum_t H(C_t|C_<t) = log N.

Part 2  factorization_distance.csv
        Structural difference between every PAIR of tokenizers of the same
        dataset.  Several separate quantities are reported; no weighted
        composite index is invented.  Different-K pairs are never compared by
        raw Hamming distance on the SID vectors.

Part 3  factorization_vs_error_diversity.csv
        Descriptive association (per dataset) between the structural distances
        and the retrieval-disagreement quantities from the two-model study.
        Sample sizes are small, so these are descriptive correlations only.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import DATASETS, FAMILY_LABELS, K_LIST, OUT, load_sid, model_name  # noqa: E402


# ---------------------------------------------------------------------------
# exact entropies of the tokenizer mapping
# ---------------------------------------------------------------------------
def cond_entropies(sid):
    """Exact H(C_t | C_<t) per position, by prefix grouping (codes are < 256).

    H(C_t | C_<t) = H(C_<=t) - H(C_<t).  Both entropies are over ITEMS, so the
    marginal must be counted on the item-indexed `keys` array -- counting the
    distinct (key, code) combinations instead silently yields negative
    conditional entropies.
    """
    N, L = sid.shape
    out = []
    for t in range(L):
        if t == 0:
            keys = np.zeros(N, dtype=np.int64)
        else:
            keys = np.zeros(N, dtype=np.int64)
            for j in range(t):
                keys = keys * 256 + sid[:, j]
        comb = keys * 256 + sid[:, t]
        _, c = np.unique(comb, return_counts=True)
        pc = c / c.sum()
        Hc = float(-(pc * np.log(pc)).sum())
        _, kc = np.unique(keys, return_counts=True)
        pk = kc / kc.sum()
        Hk = float(-(pk * np.log(pk)).sum())
        out.append(Hc - Hk)
    return np.array(out)


def marginal_entropies(sid):
    N, L = sid.shape
    out = []
    for t in range(L):
        _, c = np.unique(sid[:, t], return_counts=True)
        p = c / c.sum()
        out.append(float(-(p * np.log(p)).sum()))
    return np.array(out)


def entropy_table():
    rows = []
    profiles = {}
    for ds in DATASETS:
        for family in FAMILY_LABELS:
            for k in K_LIST:
                sid = load_sid(ds, family, k)
                L = k + 1
                cond = cond_entropies(sid)
                marg = marginal_entropies(sid)
                base_prefix = sid[:, :k]
                n_prefix_unique = len({tuple(r) for r in base_prefix})
                cum = np.cumsum(cond)
                profiles[(ds, family, k)] = {'cond': cond, 'marg': marg,
                                             'L': L, 'k': k}
                for t in range(L):
                    rows.append({
                        'dataset': ds, 'family': family, 'K': k, 'L': L,
                        'position': t,
                        'token_type': 'collision' if t == L - 1 else 'semantic',
                        'conditional_entropy_H_given_prev': cond[t],
                        'marginal_entropy_H': marg[t],
                        'cumulative_entropy_through_t': cum[t],
                        'n_unique_tokens': int(len(np.unique(sid[:, t]))),
                    })
                sem = float(cond[:L - 1].sum())
                dis = float(cond[L - 1])
                rows.append({
                    'dataset': ds, 'family': family, 'K': k, 'L': L,
                    'position': -1, 'token_type': 'summary',
                    'identity_entropy_total': float(cond.sum()),
                    'identity_entropy_log_N': float(np.log(len(sid))),
                    'semantic_conditional_entropy': sem,
                    'disambiguation_conditional_entropy': dis,
                    'semantic_fraction': sem / cond.sum(),
                    'disambiguation_fraction': dis / cond.sum(),
                    'n_items': int(len(sid)),
                    'n_unique_base_prefixes': int(n_prefix_unique),
                    'prefix_uniqueness': n_prefix_unique / len(sid),
                })
    return pd.DataFrame(rows), profiles


# ---------------------------------------------------------------------------
# pairwise structural distances
# ---------------------------------------------------------------------------
def prefix_labels(sid, k):
    """Collision-group (= base semantic prefix) label of every item, 0..G-1."""
    if k <= 0:
        return np.zeros(len(sid), dtype=np.int64)
    keys = np.zeros(len(sid), dtype=np.int64)
    for j in range(k):
        keys = keys * 256 + sid[:, j]
    _, inv = np.unique(keys, return_inverse=True)
    return inv.astype(np.int64)


def _entropy_nats(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _mutual_info_nats(a, b):
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    n = len(a)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), 1.0)
    joint /= n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def variation_of_information_nats(a, b):
    """VI(A,B) = H(A) + H(B) - 2 I(A;B), in nats."""
    return _entropy_nats(a) + _entropy_nats(b) - 2.0 * _mutual_info_nats(a, b)


def partition_metrics(labels_a, labels_b):
    return {
        'ari': float(adjusted_rand_score(labels_a, labels_b)),
        'nmi': float(normalized_mutual_info_score(labels_a, labels_b)),
        'variation_of_information': float(
            variation_of_information_nats(labels_a, labels_b)),
    }


def pair_disagreement(labels_a, labels_b, n_pairs=400000, seed=20260912):
    """P(same under exactly one partition) over uniformly sampled item pairs."""
    n = len(labels_a)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    m = i != j
    i, j = i[m], j[m]
    sa, sb = labels_a[i] == labels_a[j], labels_b[i] == labels_b[j]
    both = int((sa & sb).sum())
    only = int((sa ^ sb).sum())
    return {'pair_disagreement': only / len(i), 'pair_both_same': both / len(i)}


def structural_distance(ds, fa, ka, fb, kb, profiles):
    sa, sb = load_sid(ds, fa, ka), load_sid(ds, fb, kb)
    pa, pb = profiles[(ds, fa, ka)], profiles[(ds, fb, kb)]
    La, Lb = pa['L'], pb['L']

    # base-prefix (semantic) partition of the items -- the "collision group"
    # partition, i.e. the co-clustering the tokenizer induces.
    lab_a = prefix_labels(sa, ka)
    lab_b = prefix_labels(sb, kb)
    pm = partition_metrics(lab_a, lab_b)
    pd_ = pair_disagreement(lab_a, lab_b)

    # first-code partition
    fc = partition_metrics(sa[:, 0], sb[:, 0])

    # conditional-entropy profile distance: cumulative information curves are
    # comparable across K because both start at the first semantic token.
    m = min(ka, kb)
    cum_a = np.cumsum(pa['cond'])[:m]
    cum_b = np.cumsum(pb['cond'])[:m]
    return {
        'base_prefix_ari': pm['ari'],
        'base_prefix_nmi': pm['nmi'],
        'base_prefix_variation_of_information': pm['variation_of_information'],
        'base_prefix_pair_disagreement': pd_['pair_disagreement'],
        'base_prefix_pairs_both_same': pd_['pair_both_same'],
        'first_code_ari': fc['ari'],
        'first_code_nmi': fc['nmi'],
        'cum_semantic_info_l1': float(np.abs(cum_a - cum_b).sum()),
        'cum_semantic_info_mean_absdiff': float(np.abs(cum_a - cum_b).mean()),
        'semantic_entropy_absdiff': float(abs(
            pa['cond'][:La - 1].sum() - pb['cond'][:Lb - 1].sum())),
        'disambiguation_entropy_absdiff': float(abs(pa['cond'][La - 1] - pb['cond'][Lb - 1])),
        'total_entropy_absdiff': float(abs(pa['cond'].sum() - pb['cond'].sum())),
        'mean_cond_entropy_absdiff': float(np.abs(
            pa['cond'][:m] - pb['cond'][:m]).mean()),
        'n_prefix_groups_A': len(np.unique(lab_a)),
        'n_prefix_groups_B': len(np.unique(lab_b)),
        'K_A': ka, 'K_B': kb, 'L_A': La, 'L_B': Lb,
    }


def main():
    t0 = time.time()
    print('=== entropies ===', flush=True)
    ent, profiles = entropy_table()
    ent.to_csv(OUT / 'identity_entropy_all_factorizations.csv', index=False)
    print(f'wrote identity_entropy_all_factorizations.csv ({len(ent)} rows)')

    # identity entropy conservation check across K, within family
    chk = ent[ent['position'] == -1].copy()
    print(chk.groupby(['dataset', 'family'])['identity_entropy_total'].agg(
        ['min', 'max']).to_string())

    print('=== structural distances ===', flush=True)
    rows = []
    for ds in DATASETS:
        facts = [(f, k) for f in FAMILY_LABELS for k in K_LIST]
        for (fa, ka), (fb, kb) in itertools.combinations(facts, 2):
            if (fa, ka) == (fb, kb):
                continue
            if fa == fb and ka == kb:
                continue
            d = structural_distance(ds, fa, ka, fb, kb, profiles)
            d.update({
                'dataset': ds,
                'model_A': model_name(fa, ka), 'model_B': model_name(fb, kb),
                'family_A': fa, 'family_B': fb, 'K_A': ka, 'K_B': kb,
                'same_family': int(fa == fb), 'same_K': int(ka == kb),
                'diversity_type': ('G' if fa == fb else 'P' if ka == kb else 'GP'),
            })
            rows.append(d)
        # identical tokenizer (seed pairs) -> structural distance is 0
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'factorization_distance.csv', index=False)
    print(f'wrote factorization_distance.csv ({len(df)} rows)  [{time.time()-t0:.0f}s]')

    # ---------------- join with retrieval disagreement ----------------
    pairs_path = OUT / 'pairwise_diversity_results.csv'
    if not pairs_path.exists():
        print('pairwise_diversity_results.csv missing; skipping the join')
        return 0
    pw = pd.read_csv(pairs_path)
    key = ['dataset', 'family_A', 'family_B', 'K_A', 'K_B']
    # only canonical-seed pairs have a tokenizer-level structural distance
    can = pw[pw['seed_A'] == 2025]
    can = can[can['seed_B'] == 2025]
    merged = can.merge(df, on=key, how='inner', suffixes=('', '_sd'))
    print(f'joined {len(merged)} pair rows')

    dist_cols = ['base_prefix_ari', 'base_prefix_nmi',
                 'base_prefix_variation_of_information',
                 'base_prefix_pair_disagreement', 'first_code_ari',
                 'cum_semantic_info_l1', 'cum_semantic_info_mean_absdiff',
                 'semantic_entropy_absdiff', 'disambiguation_entropy_absdiff',
                 'mean_cond_entropy_absdiff']
    err_cols = ['hit_disagreement', 'candidate_jaccard', 'fusion_gain',
                'oracle_gain', 'oracle_exploited_fraction', 'fusion_ndcg20',
                'phi']
    out_rows = []
    for ds, g in merged.groupby('dataset'):
        for dc in dist_cols:
            for ec in err_cols:
                x = g[dc].to_numpy(dtype=float)
                y = g[ec].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 4 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
                    continue
                pr = pearsonr(x[ok], y[ok])
                sr = spearmanr(x[ok], y[ok])
                out_rows.append({
                    'dataset': ds, 'structural_distance': dc,
                    'retrieval_quantity': ec, 'n_pairs': int(ok.sum()),
                    'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
                    'spearman_rho': float(sr.statistic), 'spearman_p': float(sr.pvalue),
                    'note': 'descriptive only; small n; no significance claimed',
                })
    od = pd.DataFrame(out_rows)
    od.to_csv(OUT / 'factorization_vs_error_diversity.csv', index=False)
    print(f'wrote factorization_vs_error_diversity.csv ({len(od)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
