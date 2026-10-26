#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE B (representation side) — prefix structure and embedding-neighborhood
preservation.

For each dataset x K:

  11. prefix uniqueness / collision rate, collision-group size distribution,
      entropy H(prefix) and effective number of prefixes exp(H)
  12. embedding-neighborhood preservation: top-M neighbours by cosine similarity
      in the ORIGINAL embedding space vs top-M by SID similarity, for three SID
      similarity definitions (shared semantic tokens, Hamming, longest matching
      prefix)
  13. distance correlation over >=100,000 sampled item pairs

The final collision token is NEVER used as a metric (its uniqueness is a design
guarantee, not a finding). Only the semantic prefix is analysed.

Item row convention: the pipeline-internal convention only. No ASIN / category /
text is read or joined.

Outputs -> outputs/mechanism/phaseB_tradeoff/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.per_user import DATASETS, K_LIST, AFFINITY, sha256_file  # noqa: E402
from src.utils import repo_paths as rp                                    # noqa: E402

OUT = REPO / 'outputs' / 'mechanism' / 'phaseB_tradeoff'
MS = (10, 20, 50)
N_PAIRS = 200_000
SIM_CHUNK = 512


def sid_array(dataset, k):
    p = rp.ARTIFACT_ROOT / 'data' / dataset / f'{dataset}_t5_spectral_{AFFINITY}_v3_k{k}_sid{k+1}.npy'
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=True), p


def base_sid(dataset, k):
    p = rp.ARTIFACT_ROOT / 'data' / dataset / f'{dataset}_t5_spectral_{AFFINITY}_v3_k{k}_sid{k+1}_base_sid.npy'
    if p.exists():
        return np.load(p, allow_pickle=True), p
    a, _ = sid_array(dataset, k)
    return a[:, :k], None


def embedding(dataset):
    df = pd.read_parquet(rp.item_embedding(dataset))
    X = np.stack(df['embedding'].values).astype(np.float32)
    return X, rp.item_embedding(dataset)


def prefix_metrics(base):
    """Uniqueness, collision stats, entropy and effective prefixes."""
    n = base.shape[0]
    keys = {}
    for r in base:
        t = tuple(int(x) for x in r)
        keys[t] = keys.get(t, 0) + 1
    sizes = np.array(sorted(keys.values(), reverse=True))
    uniq = len(keys)
    p = sizes / n
    H = float(-(p * np.log(p)).sum())
    return {
        'n_items': int(n),
        'n_unique_prefix': int(uniq),
        'prefix_uniqueness': uniq / n,
        'prefix_collision_rate': 1.0 - uniq / n,
        'n_collision_groups': int((sizes > 1).sum()),
        'mean_collision_group_size': float(sizes.mean()),
        'median_collision_group_size': float(np.median(sizes)),
        'p90_collision_group_size': float(np.percentile(sizes, 90)),
        'p95_collision_group_size': float(np.percentile(sizes, 95)),
        'p99_collision_group_size': float(np.percentile(sizes, 99)),
        'max_collision_group_size': int(sizes.max()),
        'entropy_H_prefix': H,
        'effective_prefixes': float(np.exp(H)),
        'effective_prefixes_over_N': float(np.exp(H) / n),
    }


def embed_neighbors(X, M=50, chunk=SIM_CHUNK):
    """Top-M neighbours per item by cosine similarity (chunked, symmetric)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = Xn.shape[0]
    top_idx = np.full((n, M), -1, dtype=np.int32)
    top_sim = np.full((n, M), -np.inf, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = Xn[s:e] @ Xn.T                       # (b, n)
        # exclude self
        for i in range(e - s):
            S[i, s + i] = -np.inf
        idx = np.argpartition(-S, M, axis=1)[:, :M]
        sim = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sim, axis=1)
        top_idx[s:e] = np.take_along_axis(idx, order, axis=1)
        top_sim[s:e] = np.take_along_axis(sim, order, axis=1)
    return top_idx, top_sim


def neighbor_recall(base, ref_idx, M, metric, chunk=64):
    """
    For each item, take the top-M by SID similarity (excluding itself), then
    measure how many of those M are also in the item's top-M reference embedding
    neighbours. Vectorised per chunk of items.

    metric:
      'shared'  - #equal semantic tokens / K        (higher = closer)
      'hamming' - -(#differing semantic tokens)     (higher = closer)
      'lmp'     - longest matching prefix length    (higher = closer)
    """
    n, K = base.shape
    hit = np.zeros(n, dtype=np.int64)
    ref_sets = [set(ref_idx[i, :M].tolist()) for i in range(n)]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = base[s:e]                                  # (b, K)
        eq = (base[None, :, :] == blk[:, None, :])       # (b, n, K)
        if metric == 'shared':
            sim = eq.sum(axis=2) / K
        elif metric == 'hamming':
            sim = -(~eq).sum(axis=2)
        elif metric == 'lmp':
            neq = ~eq
            first = np.where(neq.any(axis=2), neq.argmax(axis=2), K)
            sim = first.astype(np.float64)
        else:
            raise ValueError(metric)
        sim = sim.astype(np.float64, copy=False)
        for i in range(e - s):
            sim[i, s + i] = -np.inf
        M_eff = min(M, n - 1)
        cand = np.argpartition(-sim, M_eff, axis=1)[:, :M_eff]
        for i in range(e - s):
            hit[s + i] = len(set(cand[i].tolist()) & ref_sets[s + i])
    return hit / M


def distance_correlation(X, base, n_pairs, rng, chunk=200_000):
    """
    Spearman / Pearson between embedding cosine similarity and SID-prefix
    similarity, over `n_pairs` randomly sampled item pairs (i != j).

    The embedding pair similarity is computed directly for the sampled pairs
    rather than read from the top-M neighbour table (which only holds the M
    nearest neighbours per item).
    """
    from scipy.stats import spearmanr, pearsonr
    n = base.shape[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    m = i != j
    i, j = i[m], j[m]
    e_sim = np.einsum('ij,ij->i', Xn[i], Xn[j])
    shared = (base[i] == base[j]).sum(axis=1) / base.shape[1]
    hamming = (base[i] != base[j]).sum(axis=1)
    out = {'n_pairs': int(len(i))}
    for nm, sv in (('shared_token', shared), ('hamming', hamming)):
        out[f'spearman_{nm}'] = float(spearmanr(e_sim, sv).statistic)
        out[f'pearson_{nm}'] = float(pearsonr(e_sim, sv)[0])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {'phase': 'B_representation_side', 'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'alignment_mode': 'canonical', 'affinity': AFFINITY,
            'n_pairs_target': N_PAIRS, 'M': list(MS), 'inputs': {}}
    pref_rows, nb_rows, dist_rows = [], [], []

    for ds in DATASETS:
        print(f'=== {ds} ===')
        X, emb_path = embedding(ds)
        prov['inputs'][f'{ds}_embedding'] = {'path': str(emb_path), 'sha256': sha256_file(emb_path)}
        # embedding reference neighbours computed ONCE per dataset
        ref_idx, ref_sim = embed_neighbors(X, M=max(MS))
        rng = np.random.default_rng(20260912)
        for k in K_LIST:
            base, bpath = base_sid(ds, k)
            if bpath:
                prov['inputs'][f'{ds}_k{k}_base_sid'] = {
                    'path': str(bpath.relative_to(REPO.parent)), 'sha256': sha256_file(bpath)}
            pm = prefix_metrics(base)
            pref_rows.append({'dataset': ds, 'K': k, **pm})
            for metric in ('shared', 'hamming', 'lmp'):
                rec = {f'neighbor_recall@{M}': float(neighbor_recall(base, ref_idx, M, metric).mean())
                       for M in MS}
                nb_rows.append({'dataset': ds, 'K': k, 'sid_metric': metric, **rec})
            dc = distance_correlation(X, base, N_PAIRS, rng)
            dist_rows.append({'dataset': ds, 'K': k, **dc})
            print(f"  K={k}: uniq={pm['prefix_uniqueness']:.4f} "
                  f"eff_prefixes/N={pm['effective_prefixes_over_N']:.4f} "
                  f"nbrR@20(shared)={[r for r in nb_rows if r['dataset']==ds and r['K']==k and r['sid_metric']=='shared'][0]['neighbor_recall@20']:.4f}")

    def w(name, rows):
        df = pd.DataFrame(rows)
        df.to_csv(OUT / name, index=False)
        print(f'  wrote {name} ({len(df)} rows)')
        return df

    w('prefix_metrics.csv', pref_rows)
    w('neighborhood_preservation.csv', nb_rows)
    w('representation_distance_correlation.csv', dist_rows)
    import subprocess
    prov['git_sha'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                     capture_output=True, text=True).stdout.strip()
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
