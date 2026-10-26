#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spectral SID construction: shape, injectivity, determinism (synthetic input).

Uses only `src/sid/` and a synthetic embedding matrix.  No dataset, no
checkpoint; artifacts are written to a temporary `CFLA_OUTPUT_ROOT`.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="cfla_test_sid_"))
os.environ["CFLA_ARTIFACT_ROOT"] = str(_TMP / "artifacts")
os.environ["CFLA_OUTPUT_ROOT"] = str(_TMP / "outputs")
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.sid.generator import generate  # noqa: E402
from src.sid.collision import add_collision_token, collision_stats  # noqa: E402

DATASET = "SynthSID"
N_ITEMS, EMB_DIM, SEED = 300, 24, 2025


def make_synthetic_embeddings(dataset: str = DATASET, n: int = N_ITEMS,
                              dim: int = EMB_DIM, seed: int = SEED) -> Path:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 6))
    X = latent @ rng.normal(size=(6, dim)) + 0.05 * rng.normal(size=(n, dim))
    ds_dir = Path(os.environ["CFLA_ARTIFACT_ROOT"]) / "data" / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "itemID": np.arange(1, n + 1),
        "embedding": [row.astype(np.float32).tolist() for row in X],
    }).to_parquet(ds_dir / "item_emb.parquet", index=False)
    np.save(ds_dir / "item_mapping.npy",
            {f"ASIN{i:05d}": i + 1 for i in range(n)}, allow_pickle=True)
    return ds_dir / "item_emb.parquet"


class TestSpectralSID(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = make_synthetic_embeddings()

    def _cfg(self, k: int, seed: int = SEED):
        return {"dataset": DATASET, "affinity": "nonneg_pearson", "k": k,
                "codebook_size": 256, "seed": seed, "pca_dim": 12,
                "item_emb_path": str(self.emb)}

    def test_shapes_and_injectivity(self):
        for k in (2, 3, 5):
            res = generate(self._cfg(k), verbose=False)
            sid = np.asarray(res["final"])
            base = np.asarray(res["base"])
            self.assertEqual(sid.shape, (N_ITEMS, k + 1))
            self.assertEqual(base.shape, (N_ITEMS, k))
            np.testing.assert_array_equal(sid[:, :k], base)
            self.assertEqual(len({tuple(r) for r in sid.tolist()}), N_ITEMS)
            self.assertTrue(sid.min() >= 0)
            self.assertTrue(sid[:, :k].max() < 256)
            self.assertTrue(res["metadata"]["final_sid_injective"])

    def test_determinism_same_seed(self):
        r1 = generate(self._cfg(3), verbose=False)
        sid1 = np.asarray(r1["final"])
        r2 = generate(self._cfg(3), verbose=False)
        sid2 = np.asarray(r2["final"])
        np.testing.assert_array_equal(sid1, sid2)

    def test_written_artifacts_and_manifest(self):
        meta = generate(self._cfg(2), verbose=False)
        d = Path(meta["out_dir"])
        for name in ("sid.npy", "base_sid.npy", "dim_groups.npy",
                     "dim_labels.npy", "metadata.json", "manifest.sha256"):
            self.assertTrue((d / name).exists(), f"missing {name}")
        self.assertNotIn(str(Path(os.environ["CFLA_ARTIFACT_ROOT"])), str(d))

    def test_collision_token_resolution(self):
        base = np.array([[1, 2], [1, 2], [3, 4], [1, 2], [5, 6]])
        stats = collision_stats(base)
        self.assertEqual(stats["num_collision_groups"], 1)
        final = add_collision_token(base, codebook_size=256)
        self.assertEqual(final.shape, (5, 3))
        np.testing.assert_array_equal(final[:, :2], base)
        self.assertEqual(len({tuple(r) for r in final.tolist()}), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
