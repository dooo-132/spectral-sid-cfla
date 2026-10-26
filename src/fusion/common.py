#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion audit utilities.

Builds the canonical union tensor used by model/fuse_beam_outputs_dynamic.py
(verified element-wise identical to it) and exposes fast evaluation of any
per-user weight matrix, so a large (gamma, eta) grid can be swept cheaply.

Read-only with respect to experiment data.
"""
import json
import numpy as np
from pathlib import Path

# Raw experiment artifacts live in the frozen audited repository, NOT here.
# Resolved through src.utils.repo_paths so the location is configurable and the
# clean repo never writes back. See docs/DATA_FORMAT.md.
try:
    from ..utils import repo_paths as _rp
except ImportError:  # allow direct script execution
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import repo_paths as _rp

ROOT = _rp.ARTIFACT_ROOT
K_LIST = [2, 3, 4, 5]
N_K = len(K_LIST)

# ---------------------------------------------------------------------------
# item-id convention
# ---------------------------------------------------------------------------
# Every beam dump in this repository was produced by the PRE-FIX
# `dump_beam_outputs.build_sid_to_item`, which stored the **0-based row index**
# of the SID array (`mp[key] = idx`).  The rest of the pipeline (parquet
# `history`/`target`, `dataset.item2code`, the SID arrays themselves) is
# 1-based: SID-array row `i` is item `i+1`.
#
# Independent evidence (frozen experiment records, not shipped):
#   * `dump.target_item_ids + 1 == parquet target` for 100% of users on all four
#     datasets (uniform shift, verified element-wise);
#   * reconstructing tokens as `sid_array[dump_item_id][pos] + 256*pos + 1`
#     reproduces `candidate_tokens` for 100.00% of candidates
#     (the naive `sid_array[dump_item_id - 1]` reproduces only ~13%).
#
# Consequence: the dumps decode item `x` where the canonical item is `x + 1`,
# and the single user whose true target is item 1 has target 0, which the dumps
# mark as invalid.  We therefore canonicalise on load.  Because the shift is
# uniform over candidates *and* targets, ranking metrics (Recall/NDCG) are
# invariant; the canonicalisation matters for collision/prefix analysis, for
# reporting item ids, and for joining against the SID arrays.
#
# `-1` is the "no valid item" sentinel emitted by `decode_tokens_to_item`.
ITEM_ID_SHIFT = 1


def canonicalize_item_ids(arr):
    """0-based dumped item ids -> canonical 1-based item ids (keep -1 sentinel)."""
    a = np.asarray(arr, dtype=np.int64)
    out = np.where(a >= 0, a + ITEM_ID_SHIFT, -1)
    return out


def de_dump(dump):
    """Return (candidate_item_ids, target_item_ids) in canonical 1-based form."""
    return (
        canonicalize_item_ids(dump["candidate_item_ids"]),
        canonicalize_item_ids(dump["target_item_ids"]),
    )

DATASETS = [
    "Beauty",
    "Cell_Phones_and_Accessories",
    "Health_and_Personal_Care",
    "Toys_and_Games",
]
AFFINITIES = ["nonneg_pearson_v3", "abspearson_v3"]


def dump_dir(dataset, affinity):
    return ROOT / f"model/beam_dumps_{dataset}_{affinity}"


def _row_softmax(scores, valid):
    """Row-wise softmax over valid candidates; invalid -> 0. Matches load_inputs()."""
    x = np.where(valid, scores, -1e30).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x) * valid
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def load_split(dataset, affinity, split):
    """Load the 4 beam dumps of one (dataset, affinity, split)."""
    d = dump_dir(dataset, affinity)
    out = []
    for k in K_LIST:
        p = d / f"{split}_k{k}_sid{k+1}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        out.append(np.load(p, allow_pickle=True))
    return out


def build_union(dumps):
    """
    Canonical union construction (identical to build_union_tensor):
      - items_pad[i, p] = item id
      - score_tensor[i, p, m] = sum of model m's softmax prob for that item
    Returns items_pad, score_tensor, target, target_pos, valid_mask.

    Item ids are canonicalised to 1-based; this is a uniform relabelling so the
    resulting rankings are unchanged relative to the raw dumps.
    """
    n, B = dumps[0]["candidate_item_ids"].shape
    target = canonicalize_item_ids(dumps[0]["target_item_ids"])

    cand = [canonicalize_item_ids(d["candidate_item_ids"]) for d in dumps]
    probs = []
    for d in dumps:
        c = np.asarray(d["candidate_item_ids"], dtype=np.int64)
        s = np.asarray(d["sequence_scores"], dtype=np.float64)
        probs.append(_row_softmax(s, c >= 0))

    umax = N_K * B
    items_pad = np.full((n, umax), -1, dtype=np.int64)
    score_tensor = np.zeros((n, umax, N_K), dtype=np.float64)

    for i in range(n):
        pos = {}
        p = 0
        for m in range(N_K):
            ci = cand[m][i]
            for j in range(B):
                it = int(ci[j])
                if it < 0:
                    continue
                if it not in pos:
                    pos[it] = p
                    items_pad[i, p] = it
                    p += 1
                score_tensor[i, pos[it], m] += probs[m][i, j]

    target_pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        w = np.where(items_pad[i] == target[i])[0]
        if len(w):
            target_pos[i] = w[0]

    valid_mask = items_pad >= 0
    return items_pad, score_tensor, target, target_pos, valid_mask


def structure_q(dataset, affinity):
    """Raw Q_K from the structure weights JSON (sorted by k)."""
    d = dump_dir(dataset, affinity)
    sj = d / f"structure_weights_{dataset}_{affinity}.json"
    if not sj.exists():
        cands = sorted(d.glob("structure_weights*.json"))
        if not cands:
            return None
        sj = cands[0]
    obj = json.loads(Path(sj).read_text())
    recs = sorted(obj["records"], key=lambda r: int(r["k"]))
    return np.array([float(r["Q"]) for r in recs], dtype=np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


def confidence_matrix(dumps):
    """
    conf = 1 - H / log(max(#valid,2)), clipped to [0,1].

    NOTE: the beam `sequence_scores` are log-probabilities, NOT probabilities.
    They must be softmax-normalised per row first; feeding raw (negative) scores
    into `p*log(p)` yields NaN and silently corrupts every fused ranking.
    """
    n = dumps[0]["candidate_item_ids"].shape[0]
    conf = np.zeros((n, N_K), dtype=np.float64)
    for m, d in enumerate(dumps):
        c = np.asarray(d["candidate_item_ids"], dtype=np.int64)
        s = np.asarray(d["sequence_scores"], dtype=np.float64)
        v = c >= 0
        pr = _row_softmax(s, v)          # <-- normalise first
        pr = np.where(v, pr, 0.0)
        ent = -np.sum(pr * np.log(pr + 1e-12), axis=1)
        vn = v.sum(axis=1)
        conf[:, m] = np.clip(1.0 - ent / (np.log(np.maximum(vn, 2)) + 1e-12), 0.0, 1.0)
    assert np.isfinite(conf).all(), "confidence matrix contains non-finite values"
    return conf


def weights_from(gamma, eta, q_norm, conf):
    """alpha_k(u) = softmax(gamma * Q_k + eta * Conf_k(u))."""
    logits = gamma * q_norm[None, :] + eta * conf
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / (w.sum(axis=1, keepdims=True) + 1e-12)


def per_user_ndcg(items_pad, score_tensor, target_pos, valid_mask, weights, topk=(5, 10, 20)):
    """
    Per-user NDCG@k and Recall@k for a per-user weight matrix.

    Users whose ground-truth target is absent from the union candidate pool
    (target_pos < 0) contribute 0.  With the default beam-30 dumps this happens
    for at most one user per dataset (the user whose true target is the first
    item, which the pre-fix dumps encode as the invalid sentinel 0).
    """
    n = items_pad.shape[0]
    weights = np.asarray(weights, dtype=np.float64)
    assert np.isfinite(weights).all(), "non-finite fusion weights"
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6), "weights must sum to 1"
    fused = np.einsum("nik,nk->ni", score_tensor, weights)
    assert np.isfinite(fused).all(), "non-finite fused scores"
    out = {}
    rows = np.where(target_pos >= 0)[0]
    hits = np.zeros(n, dtype=bool)
    ranks = np.full(n, -1, dtype=np.int64)
    if len(rows):
        ts = fused[rows, target_pos[rows]]
        r = ((fused[rows] > ts[:, None]) & valid_mask[rows]).sum(axis=1) + 1
        ranks[rows] = r
    for k in topk:
        hit = (ranks >= 1) & (ranks <= k)
        hits |= hit
        nd = np.zeros(n)
        nd[hit] = 1.0 / np.log2(ranks[hit] + 1)
        rc = np.zeros(n)
        rc[hit] = 1.0
        out[f"NDCG@{k}"] = nd
        out[f"Recall@{k}"] = rc
    out["_target_present"] = (target_pos >= 0).astype(np.float64)
    out["_rank"] = ranks.astype(np.float64)
    return out


def summarize(pu):
    return {
        f"{m}@{k}": float(pu[f"{m}@{k}"].mean())
        for m in ("Recall", "NDCG")
        for k in (5, 10, 20)
    }


def per_user_curve(pu):
    """Per-user vectors used for paired bootstrap tests."""
    return {
        f"{m}@{k}": pu[f"{m}@{k}"]
        for m in ("Recall", "NDCG")
        for k in (5, 10, 20)
    }


def evaluate_uniform(items_pad, score_tensor, target_pos, valid_mask):
    n = items_pad.shape[0]
    w = np.ones((n, N_K), dtype=np.float64) / N_K
    return summarize(per_user_ndcg(items_pad, score_tensor, target_pos, valid_mask, w))
