#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic, CPU-only end-to-end demo of the CFLA release.

Everything in this demo is **synthetic**: random item embeddings, random
candidate items and random per-factorization NLLs.  No dataset, checkpoint or
experiment artifact is used.  It demonstrates (and asserts) the two building
blocks of the paper:

  1. spectral SID construction (``src/sid/``): Pearson dimension affinity ->
     spectral partition -> per-subspace k-means (codebook 256) -> deterministic
     disambiguation token -> injective item->SID map;
  2. CFLA scoring (``src/cfla/multifactor_fusion/mf_core.py``): validation-fitted
     zNLL calibration -> log-sum-exp aggregation over factorizations -> ranking
     and Recall@K / NDCG@K.

Run:

    python3 examples/synthetic_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# The path roots are read by src.utils.repo_paths at import time, so point them
# at a throw-away demo area *before* importing any project module.
DEMO_ROOT = Path(tempfile.mkdtemp(prefix="cfla_synthetic_demo_"))
os.environ["CFLA_ARTIFACT_ROOT"] = str(DEMO_ROOT / "artifacts")
os.environ["CFLA_OUTPUT_ROOT"] = str(DEMO_ROOT / "outputs")
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.sid.generator import generate as generate_sid  # noqa: E402
from src.cfla.multifactor_fusion import mf_core  # noqa: E402
from src.cfla.multifactor_fusion.mf_common import FLABEL, M5  # noqa: E402

DATASET = "SynthDemo"
N_ITEMS = 400
EMB_DIM = 32
SEED = 2025
# The released reference scoring module (`mf_core`) labels the five-factor pool
# M5 = Spectral-K2/K3/K4/K5 + RQ-VAE (three-level reference tokenizer).  The
# CFLA math is tokenizer-agnostic: only the NLL matrix shape matters.
FACTORIZATIONS = list(M5)
N_USERS, N_CAND = 40, 30


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def build_synthetic_artifacts() -> Path:
    banner("0. synthetic input artifacts")
    rng = np.random.default_rng(SEED)
    ds_dir = Path(os.environ["CFLA_ARTIFACT_ROOT"]) / "data" / DATASET
    ds_dir.mkdir(parents=True, exist_ok=True)
    latent = rng.normal(size=(N_ITEMS, EMB_DIM // 8))
    proj = rng.normal(size=(EMB_DIM // 8, EMB_DIM))
    X = latent @ proj + 0.1 * rng.normal(size=(N_ITEMS, EMB_DIM))
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    item_mapping = {f"ASIN{i:05d}": i + 1 for i in range(N_ITEMS)}
    np.save(ds_dir / "item_mapping.npy", item_mapping, allow_pickle=True)
    pd.DataFrame({
        "itemID": np.arange(1, N_ITEMS + 1),
        "embedding": [row.astype(np.float32).tolist() for row in X],
    }).to_parquet(ds_dir / "item_emb.parquet", index=False)
    print(f"[synthetic] {N_ITEMS} items x {EMB_DIM} dims  (correlated, seed {SEED})")
    print(f"[synthetic] {ds_dir / 'item_emb.parquet'}")
    return ds_dir


def demo_sid_construction() -> None:
    banner("1. spectral SID construction (src/sid/)")
    for k in (2, 3, 4):
        cfg = {"dataset": DATASET, "affinity": "nonneg_pearson", "k": k,
               "codebook_size": 256, "seed": SEED, "pca_dim": 16,
               "item_emb_path": str(Path(os.environ["CFLA_ARTIFACT_ROOT"]) /
                                    "data" / DATASET / "item_emb.parquet")}
        res = generate_sid(cfg, verbose=False)
        sid = np.asarray(res["final"])
        base = np.asarray(res["base"])
        meta = res["metadata"]
        assert sid.shape == (N_ITEMS, k + 1), sid.shape
        assert base.shape == (N_ITEMS, k), base.shape
        assert meta["final_sid_injective"] is True
        assert np.array_equal(sid[:, :k], base)          # semantic prefix intact
        print(f"  K={k}: L={k + 1}  group sizes={meta['group_sizes']}  "
              f"injective final SID={meta['final_sid_injective']}")

    # determinism: the same seed must reproduce the identical mapping
    cfg = {"dataset": DATASET, "affinity": "nonneg_pearson", "k": 3,
           "codebook_size": 256, "seed": SEED, "pca_dim": 16,
           "item_emb_path": str(Path(os.environ["CFLA_ARTIFACT_ROOT"]) /
                                "data" / DATASET / "item_emb.parquet")}
    sid_a = np.asarray(generate_sid(cfg, verbose=False)["final"])
    sid_b = np.asarray(generate_sid(cfg, verbose=False)["final"])
    assert np.array_equal(sid_a, sid_b), "SID generation is not deterministic"
    print("  determinism: same seed reproduces an identical SID array")
    print("  PASS: SID shapes / injectivity / prefix preservation / determinism")


def build_synthetic_nll_frames():
    rng = np.random.default_rng(SEED + 7)

    def frame(n_users, n_cand):
        rows = n_users * n_cand
        users = np.repeat(np.arange(1, n_users + 1), n_cand)
        cand = rng.integers(1, N_ITEMS + 1, size=rows)
        cols = {"user": users, "candidate_item": cand}
        quality = rng.normal(size=(n_users, N_ITEMS + 1))
        for f in FACTORIZATIONS:
            base_nll = 6.0 + (0.0 if f.startswith("spectral") else 1.5)
            noise = rng.normal(scale=1.0, size=rows)
            cols[f"TotalNLL_{FLABEL[f]}"] = (
                base_nll + 1.5 * quality[users - 1, cand] + noise)
        target = rng.integers(1, N_ITEMS + 1, size=n_users)
        return pd.DataFrame(cols), target

    return frame(30, N_CAND), frame(N_USERS, N_CAND)


def independent_cfla_reference(nll_matrix: np.ndarray, scalers) -> np.ndarray:
    z = (nll_matrix - scalers[0]) / scalers[1]
    x = -z
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).ravel()


def demo_cfla_scoring() -> None:
    banner("2. CFLA: validation-fitted zNLL + log-sum-exp aggregation")
    (valid_df, _), (test_df, target) = build_synthetic_nll_frames()

    scalers = mf_core.fit_scalers(valid_df, FACTORIZATIONS)
    test_df = mf_core.apply_scalers(test_df, scalers)
    Z = mf_core.z_of(test_df, FACTORIZATIONS)
    score = mf_core.score_lse_from_znll(Z)

    nll_mat = np.stack([test_df[f"TotalNLL_{FLABEL[f]}"].values
                        for f in FACTORIZATIONS], axis=1)
    ref = independent_cfla_reference(
        nll_mat, (np.array([scalers[f][0] for f in FACTORIZATIONS]),
                  np.array([scalers[f][1] for f in FACTORIZATIONS])))
    diff = float(np.max(np.abs(score - ref)))
    print(f"  calibrated on 30x{N_CAND} SYNTHETIC validation candidates, "
          f"applied to {N_USERS}x{N_CAND} test candidates")
    print(f"  |released LSE - independent reference|_max = {diff:.3e}")
    assert diff < 1e-10, diff

    for f in FACTORIZATIONS:
        mu, sd = scalers[f]
        v = valid_df[f"TotalNLL_{FLABEL[f]}"].values
        assert abs(v.mean() - mu) < 1e-9 and abs(v.std(ddof=0) - sd) < 1e-9
    print("  PASS: calibration statistics are validation-fitted and match; "
          "aggregation matches the independent LSE reference")

    order = np.lexsort((test_df["candidate_item"].values,
                        test_df["user"].values))
    users = test_df["user"].values[order]
    items = test_df["candidate_item"].values[order]
    score = score[order]
    _, starts = np.unique(users, return_index=True)
    ends = np.append(starts[1:], len(users))
    res = mf_core.rank_and_metrics(score, items=items, target=target,
                                   starts=starts, ends=ends)
    print(f"  SYNTHETIC ranking metrics: "
          f"R@10={res['metrics']['Recall@10']:.3f}  "
          f"N@10={res['metrics']['NDCG@10']:.3f}  "
          f"target-in-union={res['in_union'].mean():.2f}")

    probs = {f: np.full_like(score, 1.0 / N_CAND) for f in FACTORIZATIONS}
    uni = np.mean([probs[f] for f in FACTORIZATIONS], axis=0)
    res_u = mf_core.rank_and_metrics(uni, items=items, target=target,
                                     starts=starts, ends=ends)
    print(f"  SYNTHETIC uniform fusion:  "
          f"R@10={res_u['metrics']['Recall@10']:.3f}  "
          f"N@10={res_u['metrics']['NDCG@10']:.3f}")
    print("  NOTE: these numbers are from random synthetic scores; they illustrate "
          "the code path, not the paper's results.")


def main() -> int:
    print("CFLA release - synthetic CPU demo (no dataset, no checkpoints)")
    print(f"demo scratch dir: {DEMO_ROOT}")
    build_synthetic_artifacts()
    demo_sid_construction()
    demo_cfla_scoring()
    banner("DONE - all synthetic assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
