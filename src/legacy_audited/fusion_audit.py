#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for the historical fusion helper module.

The historical fusion library (`fusion_audit`) is superseded in this release by
`src.fusion.common`, which provides the same helpers with configurable paths.
The paper-pipeline drivers import the helpers under the historical module name,
so this module re-exports them.
"""
from src.fusion.common import (  # noqa: F401
    DATASETS, AFFINITIES, K_LIST, N_K, ITEM_ID_SHIFT, dump_dir, load_split,
    build_union, structure_q, zscore, confidence_matrix, weights_from,
    per_user_ndcg, per_user_curve, summarize, evaluate_uniform, de_dump,
    canonicalize_item_ids, _row_softmax,
)
