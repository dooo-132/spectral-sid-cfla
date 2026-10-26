#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CFLA scoring math: validation-fitted zNLL + log-sum-exp aggregation.

Verifies the released primitives in
``src/cfla/multifactor_fusion/mf_core.py`` against an independent NumPy
re-implementation and checks the properties the paper relies on:

* calibration statistics are fitted on the validation candidates only;
* the aggregation is ``logsumexp_F(-z_F)`` with ``z_F = (NLL_F - mu_F)/sigma_F``;
* the score is monotone in each factorization's NLL;
* the ground-truth target is not used by the scoring path.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="cfla_test_math_"))
os.environ["CFLA_ARTIFACT_ROOT"] = str(_TMP / "artifacts")
os.environ["CFLA_OUTPUT_ROOT"] = str(_TMP / "outputs")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.cfla.multifactor_fusion import mf_core  # noqa: E402
from src.cfla.multifactor_fusion.mf_common import FLABEL, M5  # noqa: E402

FS = list(M5)
SEED = 2025


def make_frame(n_users=12, n_cand=7, seed=SEED, shift=0.0):
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(1, n_users + 1), n_cand)
    rows = len(users)
    df = pd.DataFrame({
        "user": users,
        "candidate_item": rng.integers(1, 50, size=rows),
    })
    for f in FS:
        df[f"TotalNLL_{FLABEL[f]}"] = 5.0 + shift + rng.normal(scale=1.2, size=rows)
    return df


def reference_znll_lse(df, scalers):
    cols = []
    for f in FS:
        mu, sd = scalers[f]
        cols.append((df[f"TotalNLL_{FLABEL[f]}"].values - mu) / sd)
    x = -np.stack(cols, axis=1)
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).ravel()


class TestCFLAMath(unittest.TestCase):
    def test_scalers_fitted_on_validation_candidates(self):
        valid = make_frame(seed=1)
        scalers = mf_core.fit_scalers(valid, FS)
        self.assertEqual(set(scalers), set(FS))
        for f in FS:
            v = valid[f"TotalNLL_{FLABEL[f]}"].values
            mu, sd = scalers[f]
            self.assertAlmostEqual(mu, float(v.mean()), places=10)
            self.assertAlmostEqual(sd, float(v.std(ddof=0)), places=10)

        test = make_frame(seed=2, shift=3.0)
        test = mf_core.apply_scalers(test, scalers)
        Z = mf_core.z_of(test, FS)
        self.assertGreater(np.abs(Z).mean(), 0.5)
        ref = reference_znll_lse(test, scalers)
        got = mf_core.score_lse_from_znll(Z)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_lse_matches_reference_and_is_monotone(self):
        valid = make_frame(seed=3)
        scalers = mf_core.fit_scalers(valid, FS)
        df = mf_core.apply_scalers(valid, scalers)
        Z = mf_core.z_of(df, FS)
        got = mf_core.score_lse_from_znll(Z)
        np.testing.assert_allclose(got, reference_znll_lse(df, scalers),
                                   rtol=0, atol=1e-12)

        df2 = df.copy()
        f0 = FS[0]
        df2[f"TotalNLL_{FLABEL[f0]}"] = df2[f"TotalNLL_{FLABEL[f0]}"] - 2.0
        Z2 = mf_core.z_of(df2, FS)
        got2 = mf_core.score_lse_from_znll(Z2)
        self.assertTrue(np.all(got2 >= got - 1e-12))

    def test_target_not_used_by_scoring(self):
        df = make_frame(seed=4)
        scalers = mf_core.fit_scalers(df, FS)
        a = mf_core.score_lse_from_znll(
            mf_core.z_of(mf_core.apply_scalers(df.copy(), scalers), FS))
        df_other_target = df.copy()
        df_other_target["target"] = np.random.default_rng(0).integers(
            1, 50, size=len(df_other_target))
        b = mf_core.score_lse_from_znll(
            mf_core.z_of(mf_core.apply_scalers(df_other_target, scalers), FS))
        np.testing.assert_array_equal(a, b)

    def test_rank_and_metrics_hand_computed(self):
        scores = np.array([0.9, 0.5, 0.1,   # user 1
                           0.2, 0.8, 0.4])  # user 2
        items = np.array([10, 20, 30, 40, 50, 60])
        target = np.array([20, 50])          # user1 -> item 20 (rank 2)
        starts = np.array([0, 3])
        ends = np.array([3, 6])
        res = mf_core.rank_and_metrics(scores, items=items, target=target,
                                       starts=starts, ends=ends)
        n2 = 1.0 / np.log2(3.0)
        self.assertAlmostEqual(res["metrics"]["NDCG@5"], (n2 + 1.0) / 2, places=12)
        self.assertAlmostEqual(res["metrics"]["Recall@5"], 1.0, places=12)
        self.assertAlmostEqual(res["metrics"]["NDCG@20"], (n2 + 1.0) / 2, places=12)
        np.testing.assert_array_equal(res["rank"], np.array([2, 1]))
        self.assertTrue(res["in_union"].all())

        target_missing = np.array([999, 999])
        res2 = mf_core.rank_and_metrics(scores, items=items,
                                        target=target_missing,
                                        starts=starts, ends=ends)
        self.assertEqual(res2["metrics"]["NDCG@5"], 0.0)
        self.assertEqual(res2["metrics"]["Recall@10"], 0.0)
        self.assertFalse(res2["in_union"].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
