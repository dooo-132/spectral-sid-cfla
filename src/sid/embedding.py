#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Item semantic embedding loading.

INPUT artifact: `data/<dataset>/item_emb.parquet`, produced upstream by
sentence-t5 encoding of item metadata. Accepts the formats the audited loader
accepted (an `embedding`/`emb`/`item_emb`/`vector` column, a single list-valued
column, or all-numeric columns).

Extracted verbatim (numerically) from
`the frozen artifact root/rqvae/generate_spectral_sid_v3.py::load_item_embedding`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_EMB_COLS = ('embedding', 'emb', 'item_emb', 'vector')
# The canonical corpora are inconsistent about this column's case:
#   Beauty / Toys        -> "ItemID"
#   Cell Phones / Health -> "itemID"
_ID_COLS = ('ItemID', 'itemID', 'item_id', 'id')

ALIGNMENT_MODES = ('canonical', 'item_id')


def find_id_column(df):
    """Return the name of the item-id column, or None."""
    for c in _ID_COLS:
        if c in df.columns:
            return c
    return None


def load_item_embedding(path) -> tuple:
    """Return (X, dataframe) with X of shape (n_items, emb_dim) float32.

    The row order is the parquet row order. Use `align_embeddings` to obtain the
    `item_id` alignment.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"item embedding parquet not found: {path}")
    df = pd.read_parquet(path)

    for col in _EMB_COLS:
        if col in df.columns:
            X = np.stack(df[col].values).astype(np.float32)
            return X, df

    if df.shape[1] == 1:
        first = df[df.columns[0]].iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            X = np.stack(df[df.columns[0]].values).astype(np.float32)
            return X, df

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 0:
        return numeric.values.astype(np.float32), df

    raise ValueError(
        f"cannot identify item embeddings in {path}; columns={list(df.columns)}")


def alignment_report(df) -> dict:
    """Describe the row -> item_id relationship without changing anything."""
    col = find_id_column(df)
    if col is None:
        return {'id_column': None, 'available': False}
    ids = np.array([int(x) for x in df[col].tolist()])
    n = len(ids)
    expected = np.arange(1, n + 1)
    mismatched = int((ids != expected).sum())
    uniq = len(set(ids.tolist()))
    return {
        'id_column': col,
        'available': True,
        'n_items': int(n),
        'n_mismatched_rows': mismatched,
        'pct_mismatched_rows': round(100.0 * mismatched / n, 6),
        'n_unique_ids': uniq,
        'id_min': int(ids.min()),
        'id_max': int(ids.max()),
        'identity_ordering': bool((ids == expected).all()),
        'is_bijection_1_to_N': bool(sorted(ids.tolist()) == list(range(1, n + 1))),
    }


def align_embeddings(X: np.ndarray, df, mode: str = 'canonical') -> tuple:
    """Return (X_used, alignment_metadata).

    mode='canonical'
        Historical behaviour: use the parquet row order as-is. Row `i` is then
        treated as item `i+1` downstream. Byte-identical to every audited run.

    mode='item_id'
        Reorder the rows so that row `i` really is item `i+1`, using the
        embedding file's own item-id column. Requires that column to be a
        bijection onto 1..N; otherwise this raises (hard fail, no silent
        fallback).

    The source parquet is never modified.
    """
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment_mode {mode!r}; "
                         f"expected one of {ALIGNMENT_MODES}")

    rep = alignment_report(df)
    meta = {'alignment_mode': mode, **rep,
            'number_rows_reordered': 0, 'percentage_rows_reordered': 0.0}

    if mode == 'canonical':
        return X, meta

    # ---- item_id ----
    if not rep['available']:
        raise ValueError(
            "alignment_mode='item_id' requires an item-id column "
            f"(looked for {_ID_COLS}) but none was found")
    if not rep['is_bijection_1_to_N']:
        raise ValueError(
            f"alignment_mode='item_id' requires item ids to be exactly 1..N "
            f"with no gaps or duplicates; got min={rep['id_min']} "
            f"max={rep['id_max']} unique={rep['n_unique_ids']} "
            f"n={rep['n_items']}")

    ids = np.array([int(x) for x in df[rep['id_column']].tolist()])
    order = np.argsort(ids, kind='stable')          # item id -> source row
    X_aligned = X[order]
    n = len(ids)
    moved = int((order != np.arange(n)).sum())
    meta.update({
        'number_rows_reordered': moved,
        'percentage_rows_reordered': round(100.0 * moved / n, 6),
        'reorder_source_rows': 'argsort(item_id)',
    })
    return X_aligned, meta
