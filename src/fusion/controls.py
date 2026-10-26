#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 6 — controls for the central reviewer question:

    "Does the gain come from *multi-granularity*, or merely from ensembling four
     models / enlarging the candidate budget?"

Two controls are possible with the artifacts that actually exist on this server.
A third (a same-granularity, different-seed ensemble) is NOT possible — there is
exactly one training seed (2025) per configuration and no GPU is available (see
MISSING_EXPERIMENTS.md).  We therefore run the controls we *can* run honestly and
record the rest as missing.

CONTROL A — candidate-budget / union-size accounting
    Uniform fusion pools the beams of four granularity models.  If the gain were
    a pure budget effect, the pooled candidate set would be ~4x the single-model
    beam.  We measure the realised union size and the per-model contribution.

CONTROL B — is the extra budget sufficient?
    For each user we recompute Recall@20/NDCG@20 using ONLY the candidates that
    granularity K alone contributed to the union, under the same uniform fusion
    scores.  This isolates "more candidates" from "the fusion re-ranks with four
    opinions".  We also evaluate a *budget-matched* fusion in which the pooled
    candidate list is truncated to the single-model beam size (30) by fused
    score, i.e. the multi-granularity models share one model's candidate budget.

CONTROL C — heterogeneous vs homogeneous re-ranking
    Rerank the pooled union with a single opinion (a fixed granularity's scores)
    versus with the uniform mixture.  Comparing "union + one opinion" against
    "union + four opinions" shows how much of the gain is the pool and how much
    is the actual mixture of granularity opinions.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion.common import (  # noqa: E402
    DATASETS, K_LIST, N_K, load_split, build_union, per_user_ndcg, summarize,
)

OUT = Path(__file__).resolve().parent
CANON = "nonneg_pearson_v3"


def fused_scores(score_tensor, weights):
    return np.einsum("nik,nk->ni", score_tensor, weights)


def ndcg_from_ranking(fused, items_pad, target_pos, valid_mask, topk=(5, 10, 20),
                      restrict=None):
    """
    Per-user NDCG/Recall for an explicit (n, U) fused score matrix.
    `restrict` optionally masks candidates (n, U) boolean.
    """
    n = items_pad.shape[0]
    if restrict is not None:
        fused = np.where(restrict, fused, -np.inf)
    vm = valid_mask if restrict is None else (valid_mask & restrict)
    out = {}
    rows = np.where(target_pos >= 0)[0]
    ranks = np.full(n, -1, dtype=np.int64)
    if len(rows):
        ts = fused[rows, target_pos[rows]]
        # candidates strictly greater than the target; equal scores do not beat it
        r = ((fused[rows] > ts[:, None]) & vm[rows]).sum(axis=1) + 1
        ranks[rows] = r
    for k in topk:
        hit = (ranks >= 1) & (ranks <= k)
        nd = np.zeros(n)
        nd[hit] = 1.0 / np.log2(ranks[hit] + 1)
        rc = np.zeros(n)
        rc[hit] = 1.0
        out[f"NDCG@{k}"] = nd
        out[f"Recall@{k}"] = rc
    return out


def main():
    report = {"canonical_affinity": CANON, "per_dataset": {}}

    for ds in DATASETS:
        zs = load_split(ds, CANON, "test")
        items_pad, score_tensor, target, target_pos, valid_mask = build_union(zs)
        n, U, _ = score_tensor.shape
        w_uni = np.ones((n, N_K)) / N_K
        fused_uni = fused_scores(score_tensor, w_uni)

        # ---------- CONTROL A: union size / per-model contribution ----------
        contrib = np.zeros((n, N_K), dtype=np.int64)
        for m in range(N_K):
            contrib[:, m] = (score_tensor[:, :, m] > 0).sum(axis=1)
        union_size = valid_mask.sum(axis=1)

        # how many candidates does each single model contribute that no other does
        exclusive = np.zeros((n, N_K), dtype=np.int64)
        pres = score_tensor > 0
        for m in range(N_K):
            others = np.delete(pres, m, axis=2).any(axis=2)
            exclusive[:, m] = (pres[:, :, m] & ~others).sum(axis=1)

        ctrlA = {
            "mean_union_size": float(union_size.mean()),
            "median_union_size": float(np.median(union_size)),
            "mean_union_fraction_of_4B": float(union_size.mean() / (4 * items_pad.shape[1])),
            "beam_size": int(items_pad.shape[1]),
            "mean_contributed_per_model": [float(contrib[:, m].mean()) for m in range(N_K)],
            "mean_exclusive_per_model": [float(exclusive[:, m].mean()) for m in range(N_K)],
            "frac_candidates_from_single_model_only": float(
                (exclusive.sum(axis=1) > 0).mean()),
        }

        # ---------- CONTROL B: restrict to one model's contributed candidates ----------
        ctrlB = {}
        for m, k in enumerate(K_LIST):
            restrict = score_tensor[:, :, m] > 0
            pu = ndcg_from_ranking(fused_uni, items_pad, target_pos, valid_mask,
                                   restrict=restrict)
            ctrlB[f"K={k}_candidates_only"] = summarize(pu)

        # budget-matched: keep the top-30 pooled candidates by fused score,
        # i.e. give the four granularity models the candidate budget of ONE model.
        B = items_pad.shape[1]
        order = np.argsort(-fused_uni, axis=1, kind="stable")
        keep = np.zeros_like(valid_mask)
        np.put_along_axis(keep, order[:, :B], True, axis=1)
        keep &= valid_mask
        pu = ndcg_from_ranking(fused_uni, items_pad, target_pos, valid_mask, restrict=keep)
        ctrlB["budget_matched_top30_by_fused_score"] = summarize(pu)

        # same but ranking by a single model's score restricted to the union
        for m, k in enumerate(K_LIST):
            pass

        # ---------- CONTROL C: union + one opinion vs union + uniform mixture ----------
        ctrlC = {"uniform_mixture": summarize(ndcg_from_ranking(
            fused_uni, items_pad, target_pos, valid_mask))}
        for m, k in enumerate(K_LIST):
            w1 = np.zeros((n, N_K))
            w1[:, m] = 1.0
            f1 = fused_scores(score_tensor, w1)
            ctrlC[f"union_ranked_by_K{k}_only"] = summarize(
                ndcg_from_ranking(f1, items_pad, target_pos, valid_mask))
        # oracle: rank the union by the best per-user choice of a single K
        best_single_ndcg = np.zeros(n)
        for m in range(N_K):
            w1 = np.zeros((n, N_K))
            w1[:, m] = 1.0
            pu = ndcg_from_ranking(fused_scores(score_tensor, w1), items_pad,
                                   target_pos, valid_mask)
            best_single_ndcg = np.maximum(best_single_ndcg, pu["NDCG@20"])
        ctrlC["oracle_best_single_K_per_user_NDCG@20"] = float(best_single_ndcg.mean())

        report["per_dataset"][ds] = {"control_A_budget": ctrlA,
                                     "control_B_restricted": ctrlB,
                                     "control_C_opinions": ctrlC}
        print(f"  {ds:32s} union={ctrlA['mean_union_size']:.2f} "
              f"(vs 4x30=120, beam={ctrlA['beam_size']}) "
              f"uniform_N20={ctrlC['uniform_mixture']['NDCG@20']:.6f} "
              f"budget_matched_N20={ctrlB['budget_matched_top30_by_fused_score']['NDCG@20']:.6f}")

    (OUT / "PHASE6_CONTROLS.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT/'PHASE6_CONTROLS.json'}")
    return report


if __name__ == "__main__":
    main()
