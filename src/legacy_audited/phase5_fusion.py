#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2 (cont.) + PHASE 5 — fusion comparison, significance tests, and the
canonical result tables.

Produces, all from raw beam dumps:
  * fusion_strategy_ablation.csv     — Best Single K / Uniform / Valid Grid /
                                       Structure-aware / Confidence-only / Structure+Confidence
  * dynamic_vs_uniform_bootstrap.csv — paired user-level bootstrap (P2)
  * fusion_vs_single_bootstrap.csv   — Uniform multi-granularity vs Best Single K (section 9)
  * per-K single-granularity metrics
All metrics full precision.  No test-set hyper-parameter selection ever happens
here: the (gamma, eta) used for the "valid grid" row is selected on `valid` and
then applied unchanged to `test`.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # release repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.legacy_audited.fusion_audit import (  # noqa: E402
    DATASETS, AFFINITIES, K_LIST, N_K, load_split, build_union, structure_q,
    zscore, confidence_matrix, weights_from, per_user_ndcg, per_user_curve,
    summarize,
)

OUTDIR = Path(__file__).resolve().parent
# The historical module created a `canonical/` directory at import time.  In this
# release the module is imported as a helper library by the reproduction drivers,
# which create their outputs under CFLA_OUTPUT_ROOT, so no import-time write.
CANON = OUTDIR / "canonical"

CANONICAL_AFFINITY = "nonneg_pearson_v3"
GAMMA_GRID = [-8, -5, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 5, 8]
ETA_GRID = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]
N_BOOT = 5000
RNG_SEED = 20260912
METRICS = [("Recall", 5), ("Recall", 10), ("Recall", 20),
           ("NDCG", 5), ("NDCG", 10), ("NDCG", 20)]


def cell(dataset, affinity):
    splits = {}
    for split in ("valid", "test"):
        zs = load_split(dataset, affinity, split)
        splits[split] = build_union(zs)
    q = structure_q(dataset, affinity)
    qn = zscore(q)
    conf_v = confidence_matrix(load_split(dataset, affinity, "valid"))
    conf_t = confidence_matrix(load_split(dataset, affinity, "test"))
    return splits, qn, conf_v, conf_t


def per_user_for_weights(union, weights):
    items_pad, score_tensor, target, target_pos, valid_mask = union
    return per_user_ndcg(items_pad, score_tensor, target_pos, valid_mask, weights)


def grid_ndcg20(items_pad, score_tensor, target_pos, valid_mask, qn, conf,
                gamma_grid, eta_grid, chunk=64):
    """
    Vectorised (gamma, eta) grid evaluation of NDCG@20 on one split.

    Returns arrays of shape (len(gamma_grid), len(eta_grid)) for NDCG@20 and
    Recall@20.  Ranking is computed independently for every grid point; this is
    exactly equivalent to calling per_user_ndcg once per (gamma, eta) but ~2
    orders of magnitude faster.
    """
    n = items_pad.shape[0]
    rows = np.where(target_pos >= 0)[0]
    tpos = target_pos[rows]
    valid_rows = valid_mask[rows]
    gs = np.asarray(gamma_grid, dtype=np.float64)
    es = np.asarray(eta_grid, dtype=np.float64)
    G, E = np.meshgrid(gs, es, indexing="ij")
    G = G.ravel()
    E = E.ravel()
    n_grid = G.size
    nd = np.zeros((n_grid,), dtype=np.float64)
    rc = np.zeros((n_grid,), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sl = rows[(rows >= s) & (rows < e)]
        if sl.size == 0:
            continue
        pos = target_pos[sl]
        vm = valid_mask[sl]
        st = score_tensor[sl]                      # (b, U, K)
        cf = conf[sl]                              # (b, K)
        for gi in range(n_grid):
            logits = G[gi] * qn[None, :] + E[gi] * cf
            logits = logits - logits.max(axis=1, keepdims=True)
            w = np.exp(logits)
            w /= w.sum(axis=1, keepdims=True)
            fused = np.einsum("buk,bk->bu", st, w)
            ts = fused[np.arange(sl.size), pos]
            r = ((fused > ts[:, None]) & vm).sum(axis=1) + 1
            hit = r <= 20
            nd[gi] += float((1.0 / np.log2(r[hit] + 1)).sum())
            rc[gi] += float(hit.sum())
    nd /= max(len(rows), 1)
    rc /= max(len(rows), 1)
    return nd.reshape(len(gs), len(es)), rc.reshape(len(gs), len(es))


def uniform_weights(n):
    return np.ones((n, N_K)) / N_K


def single_k_weights(n, m):
    w = np.zeros((n, N_K))
    w[:, m] = 1.0
    return w


def select_on_valid(U_v, qn, conf):
    """Grid-search (gamma, eta) on the VALIDATION split only."""
    items_pad, score_tensor, target, target_pos, valid_mask = U_v
    nd, rc = grid_ndcg20(items_pad, score_tensor, target_pos, valid_mask, qn, conf,
                         GAMMA_GRID, ETA_GRID)
    flat_nd = nd.ravel()
    order = np.argsort(-flat_nd, kind="stable")
    best_idx = int(order[0])
    gi, ei = np.unravel_index(best_idx, nd.shape)
    table = []
    for a in range(nd.shape[0]):
        for b in range(nd.shape[1]):
            table.append({"gamma": float(GAMMA_GRID[a]), "eta": float(ETA_GRID[b]),
                          "valid_NDCG@20": float(nd[a, b]),
                          "valid_Recall@20": float(rc[a, b])})
    return {"gamma": float(GAMMA_GRID[gi]), "eta": float(ETA_GRID[ei]),
            "valid_NDCG@20": float(nd[gi, ei]),
            "valid_Recall@20": float(rc[gi, ei])}, table


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Paired user-level bootstrap over per-user metric vectors.
    Returns delta = mean(a) - mean(b), 95% CI, and a two-sided p-value
    (fraction of bootstrap replicates on the opposite side of 0, doubled).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    n = a.shape[0]
    d = a - b
    obs = float(d.mean())
    if np.allclose(d, 0.0):
        return {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": float("nan"),
                "n_users": int(n), "note": "identical vectors"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p-value: 2 * min(P(boot<=0), P(boot>=0)), capped at 1
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {"delta": obs, "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p,
            "n_users": int(n),
            "pct_change": float(100.0 * obs / b.mean()) if b.mean() != 0 else float("nan")}


def main():
    results = {"single_k": [], "fusion": [], "boot_uniform_vs_single": [],
               "boot_dynamic_vs_uniform": [], "valid_selection": []}

    for dataset in DATASETS:
        for affinity in AFFINITIES:
            is_canon = affinity == CANONICAL_AFFINITY
            splits, qn, conf_v, conf_t = cell(dataset, affinity)
            U_v, U_t = splits["valid"], splits["test"]
            n_v, n_t = U_v[0].shape[0], U_t[0].shape[0]

            # ---------- single granularity (test) ----------
            sk_pu = {}
            for m, k in enumerate(K_LIST):
                pu = per_user_for_weights(U_t, single_k_weights(n_t, m))
                sk_pu[k] = pu
                results["single_k"].append({
                    "dataset": dataset, "affinity": affinity, "k": k,
                    "split": "test", **summarize(pu),
                })

            # ---------- uniform fusion ----------
            uni_v = per_user_for_weights(U_v, uniform_weights(n_v))
            uni_t = per_user_for_weights(U_t, uniform_weights(n_t))

            # ---------- structure-aware (gamma from valid, eta = 0) ----------
            st_v = per_user_for_weights(U_v, weights_from(1.0, 0.0, qn, conf_v))
            st_t = per_user_for_weights(U_t, weights_from(1.0, 0.0, qn, conf_t))

            # ---------- confidence-only (gamma = 0, eta from valid) ----------
            tab_c = []
            for e in ETA_GRID:
                pu = per_user_for_weights(U_v, weights_from(0.0, e, qn, conf_v))
                tab_c.append({"eta": e, "valid_NDCG@20": float(pu["NDCG@20"].mean())})
            best_eta_c = max(tab_c, key=lambda r: r["valid_NDCG@20"])
            conf_only_t = per_user_for_weights(U_t, weights_from(0.0, best_eta_c["eta"], qn, conf_t))

            # ---------- structure + confidence (valid grid search) ----------
            best, table = select_on_valid(U_v, qn, conf_v)
            g_star, e_star = best["gamma"], best["eta"]
            dyn_t = per_user_for_weights(U_t, weights_from(g_star, e_star, qn, conf_t))
            results["valid_selection"].append({
                "dataset": dataset, "affinity": affinity,
                "selected_gamma": g_star, "selected_eta": e_star,
                "valid_NDCG@20": best["valid_NDCG@20"],
                "test_NDCG@20": float(dyn_t["NDCG@20"].mean()),
                "confidence_only_selected_eta": best_eta_c["eta"],
                "grid": table,
            })

            best_single_m = max(range(N_K), key=lambda m: sk_pu[K_LIST[m]]["NDCG@20"].mean())
            best_single_k = K_LIST[best_single_m]
            bs_t = sk_pu[best_single_k]

            for name, pu in [("best_single", bs_t), ("uniform", uni_t),
                             ("valid_grid_search", dyn_t), ("structure_aware", st_t),
                             ("confidence_only", conf_only_t)]:
                results["fusion"].append({
                    "dataset": dataset, "affinity": affinity, "strategy": name,
                    "is_canonical_affinity": is_canon,
                    "best_single_k": best_single_k if name == "best_single" else None,
                    **summarize(pu),
                })

            # ---------- significance: uniform vs best single K ----------
            for metric, k in METRICS:
                key = f"{metric}@{k}"
                r = paired_bootstrap(uni_t[key], bs_t[key])
                results["boot_uniform_vs_single"].append({
                    "dataset": dataset, "affinity": affinity, "metric": key,
                    "comparison": f"uniform_vs_best_single_K{best_single_k}",
                    "best_single_k": best_single_k,
                    "best_single_value": float(bs_t[key].mean()),
                    "uniform_value": float(uni_t[key].mean()),
                    "delta": r["delta"], "pct_change": r.get("pct_change"),
                    "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                    "n_users": r["n_users"], "n_boot": N_BOOT,
                    "significant_95": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0),
                    "note": r.get("note"),
                })

            # ---------- significance: dynamic (valid-selected) vs uniform ----------
            for metric, k in METRICS:
                key = f"{metric}@{k}"
                r = paired_bootstrap(dyn_t[key], uni_t[key])
                results["boot_dynamic_vs_uniform"].append({
                    "dataset": dataset, "affinity": affinity, "metric": key,
                    "comparison": "dynamic_vs_uniform",
                    "gamma": g_star, "eta": e_star,
                    "uniform_value": float(uni_t[key].mean()),
                    "dynamic_value": float(dyn_t[key].mean()),
                    "delta": r["delta"], "pct_change": r.get("pct_change"),
                    "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_value": r["p_value"],
                    "n_users": r["n_users"], "n_boot": N_BOOT,
                    "significant_95": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0),
                    "note": r.get("note"),
                })

            print(f"  {dataset:32s} {affinity:18s} best_single=K{best_single_k} "
                  f"uniform_N20={uni_t['NDCG@20'].mean():.6f} "
                  f"dyn(g={g_star},e={e_star})_N20={dyn_t['NDCG@20'].mean():.6f}")

    (OUTDIR / "PHASE5_FUSION.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUTDIR / 'PHASE5_FUSION.json'}")
    return results


if __name__ == "__main__":
    main()
