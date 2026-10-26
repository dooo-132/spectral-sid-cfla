#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CFLA: cross-factorization likelihood aggregation (runnable implementation).

This module implements the paper's inference pipeline end to end, using only
artifacts that the user produces with the shipped stages:

    scripts/prepare_data.sh        -> data/<Dataset>/{train,valid,test}.parquet
    scripts/build_sid.sh           -> SID array per factorization (or an external
                                      RQ-VAE / RQ-OPQ implementation)
    scripts/train.sh               -> one T5 checkpoint per factorization
    scripts/generate_candidates.sh -> beam dump per factorization and split
    scripts/run_cfla.sh            -> this module

Pipeline (paper Sections 3.3-3.4):

1. candidate union:   C(u) = U_F C_F(u)  (generated items only; the target is
   never inserted);
2. candidate NLL:     NLL_F(u,i) = -sum_l log P_theta_F(SID_F(i)_l | S_F(u), <l),
   teacher-forced on the complete SID under factorization F (target-free);
3. calibration:       z_F = (NLL_F - mu_F) / sigma_F, with mu_F/sigma_F fitted on
   ALL validation candidates and frozen for test;
4. aggregation:       S_CFLA = log sum_F exp(-z_F); rank descending;
5. evaluation:        item-level Recall@K / NDCG@K and paired user-level
   bootstrap over users.

The baseline scores on the same union are also available:
``uniform``  = mean of per-factorization retrieval probabilities,
``minnll``   = -min_F z_F.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def load_config(path) -> dict:
    """Load the pool config, expanding ``${VAR}``/``$VAR`` in every string.

    The three CFLA path variables fall back to the repository-relative defaults
    (``<repo>/artifacts``, ``<repo>/outputs``) when they are not set, so the
    example config works without exporting anything.
    """
    import os
    import sys
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.utils import repo_paths as rp
    os.environ.setdefault("CFLA_ARTIFACT_ROOT", str(rp.ARTIFACT_ROOT))
    os.environ.setdefault("CFLA_OUTPUT_ROOT", str(rp.OUTPUT_ROOT))
    os.environ.setdefault("CFLA_UPSTREAM_ROOT", str(rp.UPSTREAM_ROOT))
    raw = Path(path).read_text()
    cfg = yaml.safe_load(raw)
    cfg = _expand(cfg)
    if "factorizations" not in cfg or not cfg["factorizations"]:
        raise SystemExit(f"config {path} has no `factorizations` list")
    for f in cfg["factorizations"]:
        for key in ("name", "sid", "ckpt", "beam_valid", "beam_test"):
            if key not in f:
                raise SystemExit(f"factorization entry missing `{key}`: {f}")
    for key in ("dataset_dir", "codebook_size", "max_len"):
        if key not in cfg:
            raise SystemExit(f"config {path} missing `{key}`")
    return cfg


def _expand(obj):
    import os
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# beam dumps: canonicalisation, softmax, candidate union
# ---------------------------------------------------------------------------
def canonicalize_item_ids(arr):
    """0-based dumped item ids -> canonical 1-based ids (keep -1 sentinel)."""
    a = np.asarray(arr, dtype=np.int64)
    return np.where(a >= 0, a + 1, -1)


def row_softmax(scores, valid):
    x = np.where(valid, scores, -1e30).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x) * valid
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def load_dumps(factorizations, split):
    """Load one beam dump per factorization and check user alignment."""
    dumps = []
    for f in factorizations:
        p = Path(f[f"beam_{split}"])
        if not p.is_file():
            raise FileNotFoundError(f"beam dump not found: {p} ({f['name']})")
        dumps.append(np.load(p, allow_pickle=True))
    n = dumps[0]["candidate_item_ids"].shape[0]
    for f, d in zip(factorizations, dumps):
        if d["candidate_item_ids"].shape[0] != n:
            raise SystemExit(
                f"beam dumps are not user-aligned: {f['name']} has "
                f"{d['candidate_item_ids'].shape[0]} users, expected {n}")
    return dumps, n


def build_union(dumps):
    """Candidate union and per-factorization retrieval probabilities.

    Returns dict with
      items:  list of length n_users, each a 1-based item-id array (union order)
      probs:  list of length n_users, each (n_candidates, n_factorizations)
      target: (n_users,) 1-based target item id
      target_pos: (n_users,) index of the target inside its user's union, -1 if absent
    """
    n = dumps[0]["candidate_item_ids"].shape[0]
    nf = len(dumps)
    target = canonicalize_item_ids(dumps[0]["target_item_ids"])
    items, probs, target_pos = [], [], np.full(n, -1, dtype=np.int64)
    for i in range(n):
        pos = {}
        order = []
        per_factor = []
        for m, d in enumerate(dumps):
            ci = canonicalize_item_ids(d["candidate_item_ids"][i])
            valid = ci >= 0
            p = row_softmax(np.asarray(d["sequence_scores"][i], dtype=np.float64)[None, :],
                            valid[None, :])[0]
            per_factor.append((ci, p))
            for it in ci[valid]:
                it = int(it)
                if it not in pos:
                    pos[it] = len(order)
                    order.append(it)
        items_i = np.asarray(order, dtype=np.int64)
        prob_i = np.zeros((len(order), nf), dtype=np.float64)
        for m, (ci, p) in enumerate(per_factor):
            for it, pp in zip(ci, p):
                if it >= 0:
                    prob_i[pos[int(it)], m] += float(pp)
        items.append(items_i)
        probs.append(prob_i)
        w = np.where(items_i == target[i])[0]
        if len(w):
            target_pos[i] = int(w[0])
    return {"items": items, "probs": probs, "target": target,
            "target_pos": target_pos}


def flat_pairs(union):
    """Flatten (user, candidate) rows for scoring; returns user_idx, item_ids."""
    user_idx, item_ids = [], []
    for i, items in enumerate(union["items"]):
        user_idx.append(np.full(len(items), i, dtype=np.int64))
        item_ids.append(items)
    return np.concatenate(user_idx), np.concatenate(item_ids)


# ---------------------------------------------------------------------------
# history encoding and candidate NLL (teacher forcing)
# ---------------------------------------------------------------------------
def load_eval_histories(parquet_path, item_to_tokens, max_len, n_users):
    """Encode evaluation histories (one row per user) under one factorization.

    Mirrors ``src.tiger.dataset.GenRecDataset``: histories are truncated/padded to
    ``max_len`` ITEMS, then each item is replaced by its ``L`` SID tokens and
    flattened; token 0 is PAD.
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if len(df) != n_users:
        raise SystemExit(
            f"{parquet_path} has {len(df)} users but the beam dumps have "
            f"{n_users}; regenerate the dumps from this split")
    sid_len = len(next(iter(item_to_tokens.values()))) if item_to_tokens else 0
    if sid_len == 0:
        raise SystemExit("empty item->token mapping")
    out = np.zeros((n_users, max_len * sid_len), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        hist = list(row.history)[-max_len:]
        tokens = []
        for item in hist:
            tokens.extend(item_to_tokens.get(int(item), [0] * sid_len))
        tokens = tokens[-max_len * sid_len:]
        out[i, :len(tokens)] = tokens
    return out


def load_item_to_tokens(sid_path, codebook_size):
    """1-based item id -> slot-wise token array for one factorization."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from src.tiger.dataset import item2code
    item_to_code, _ = item2code(sid_path, codebook_size=codebook_size)
    return item_to_code


def candidate_nll(model, histories, user_idx, item_ids, item_to_tokens,
                  device="cpu", batch_size=256):
    """Teacher-forced negative log-likelihood of every (user, candidate) row."""
    import torch
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from src.analysis.teacher_forced import teacher_forced_logprobs

    sid_len = len(next(iter(item_to_tokens.values())))
    nll = np.empty(len(item_ids), dtype=np.float64)
    order = np.argsort(user_idx, kind="stable")   # group rows per user
    user_idx = user_idx[order]
    item_ids = item_ids[order]
    nll_sorted = np.empty(len(item_ids), dtype=np.float64)
    step = batch_size
    for s in range(0, len(item_ids), step):
        e = min(s + step, len(item_ids))
        h = np.stack([histories[u] for u in user_idx[s:e]])
        t = np.stack([np.asarray(item_to_tokens[int(it)], dtype=np.int64)
                      for it in item_ids[s:e]])
        logp = teacher_forced_logprobs(model, h, t, device=device,
                                       batch_size=step)
        nll_sorted[s:e] = -logp.sum(axis=1)
    nll[order] = nll_sorted
    return nll


def load_model(ckpt_path, device="cpu"):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from src.analysis.teacher_forced import load_model_from_ckpt
    model, vocab = load_model_from_ckpt(ckpt_path, device=device)
    return model, vocab


# ---------------------------------------------------------------------------
# calibration, aggregation, metrics
# ---------------------------------------------------------------------------
def fit_scalers(nll_by_factor):
    """mu/sigma per factorization over ALL rows of the given (validation) block."""
    scalers = {}
    for name, nll in nll_by_factor.items():
        v = np.asarray(nll, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise SystemExit(f"no finite validation NLL for factorization {name}")
        scalers[name] = (float(v.mean()), float(v.std(ddof=0)))
    return scalers


def zscore(nll, scaler):
    mu, sd = scaler
    return (np.asarray(nll, dtype=np.float64) - mu) / (sd if sd > 0 else 1.0)


def score_cfla(z_matrix):
    """logsumexp_F(-z_F) row-wise."""
    x = -np.asarray(z_matrix, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).ravel()


def score_minnll(z_matrix):
    return -np.min(z_matrix, axis=1)


def score_uniform(prob_matrix):
    return np.asarray(prob_matrix, dtype=np.float64).mean(axis=1)


def per_user_metrics(union, scores, topk=(5, 10, 20)):
    """Recall@k / NDCG@k on item rankings; targets outside the union score 0."""
    out = {f"Recall@{k}": [] for k in topk}
    out.update({f"NDCG@{k}": [] for k in topk})
    out["rank"] = []
    out["in_union"] = []
    starts = np.cumsum([0] + [len(x) for x in union["items"]])
    for i, items in enumerate(union["items"]):
        s = scores[starts[i]:starts[i + 1]]
        tp = int(union["target_pos"][i])
        if tp < 0:
            for k in topk:
                out[f"Recall@{k}"].append(0.0)
                out[f"NDCG@{k}"].append(0.0)
            out["rank"].append(0)
            out["in_union"].append(False)
            continue
        order = np.argsort(-s, kind="stable")
        ranked = items[order]
        pos = np.where(ranked == union["target"][i])[0]
        r = int(pos[0]) + 1 if len(pos) else 0
        out["rank"].append(r)
        out["in_union"].append(r > 0)
        for k in topk:
            hit = 0 < r <= k
            out[f"Recall@{k}"].append(1.0 if hit else 0.0)
            out[f"NDCG@{k}"].append(1.0 / np.log2(r + 1) if hit else 0.0)
    out["in_union"] = np.asarray(out["in_union"], dtype=bool)
    for key in list(out):
        out[key] = np.asarray(out[key])
    return out


def summarize(user_metrics, topk=(5, 10, 20)):
    return {f"{m}@{k}": float(user_metrics[f"{m}@{k}"].mean())
            for m in ("Recall", "NDCG") for k in topk}


def paired_bootstrap(a, b, n_boot=5000, seed=20260912):
    """Paired user-level bootstrap of (a - b) on per-user metric vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p), "n_boot": int(n_boot)}
