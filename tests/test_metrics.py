#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Item-level evaluation metrics: Recall@K / NDCG@K and item-id canonicalisation.

Verifies the released evaluator primitives in ``src/fusion/common.py`` against
hand-computed values on tiny synthetic inputs.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.fusion.common import (  # noqa: E402
    canonicalize_item_ids, per_user_ndcg, summarize, _row_softmax,
)


class TestMetrics(unittest.TestCase):
    def test_canonicalize_item_ids(self):
        np.testing.assert_array_equal(
            canonicalize_item_ids(np.array([-1, 0, 4, 9])),
            np.array([-1, 1, 5, 10]))

    def test_softmax_rows_ignore_invalid(self):
        scores = np.array([[0.0, -1.0, -2.0], [0.0, 0.0, 0.0]])
        valid = np.array([[True, True, True], [True, True, False]])
        p = _row_softmax(scores, valid)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-12)
        self.assertEqual(p[1, 2], 0.0)

    def test_per_user_ndcg_hand_computed(self):
        items = np.array([[10, 20, 30]])
        score_tensor = np.array([[[0.9], [0.5], [0.1]]])
        valid = np.ones((1, 3), dtype=bool)
        weights = np.ones((1, 1))

        out = per_user_ndcg(items, score_tensor, np.array([0]), valid, weights)
        self.assertEqual(out["NDCG@5"][0], 1.0)
        self.assertEqual(out["Recall@5"][0], 1.0)
        self.assertEqual(out["_rank"][0], 1.0)

        out = per_user_ndcg(items, score_tensor, np.array([2]), valid, weights)
        self.assertAlmostEqual(out["NDCG@5"][0], 0.5, places=12)
        self.assertEqual(out["Recall@5"][0], 1.0)

        out = per_user_ndcg(items, score_tensor, np.array([-1]), valid, weights)
        self.assertEqual(out["NDCG@5"][0], 0.0)
        self.assertEqual(out["Recall@5"][0], 0.0)
        self.assertEqual(out["_rank"][0], -1.0)

    def test_weights_mix_two_models(self):
        items = np.array([[1, 2, 3]])
        score_tensor = np.array([[[0.9, 0.1],
                                  [0.5, 0.5],
                                  [0.1, 0.9]]])
        self.assertEqual(score_tensor.shape, (1, 3, 2))
        valid = np.ones((1, 3), dtype=bool)
        w = np.array([[0.5, 0.5]])
        fused = np.einsum("nik,nk->ni", score_tensor, w)
        np.testing.assert_allclose(fused, [[0.5, 0.5, 0.5]])
        out = per_user_ndcg(items, score_tensor, np.array([1]), valid, w)
        self.assertEqual(out["Recall@5"][0], 1.0)

    def test_summarize_keys(self):
        items = np.array([[1, 2, 3], [4, 5, 6]])
        score_tensor = np.array([[[0.9], [0.5], [0.1]], [[0.1], [0.5], [0.9]]])
        valid = np.ones((2, 3), dtype=bool)
        weights = np.ones((2, 1))
        out = per_user_ndcg(items, score_tensor, np.array([0, 1]), valid, weights)
        s = summarize(out)
        for metric in ("Recall@5", "Recall@10", "Recall@20",
                       "NDCG@5", "NDCG@10", "NDCG@20"):
            self.assertIn(metric, s)
            self.assertTrue(0.0 <= s[metric] <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
