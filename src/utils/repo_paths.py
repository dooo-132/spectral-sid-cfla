#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path resolution for the CFLA release.

Everything is configured through environment variables.  All defaults are
relative to this repository, so a fresh clone runs without editing any file:

    REPO_ROOT      this repository (code, configs, docs).
    ARTIFACT_ROOT  INPUT artifacts that the user must provide: preprocessed
                   Amazon-Reviews-2014 splits, item embeddings/mappings, SID
                   arrays, checkpoints and beam dumps.  Read-only from the
                   code's point of view.
    OUTPUT_ROOT    every NEW artifact produced by a run is written here.

Environment variables
---------------------
    CFLA_ARTIFACT_ROOT   input artifact root        (default: <repo>/artifacts)
    CFLA_UPSTREAM_ROOT   optional fallback root     (default: $CFLA_ARTIFACT_ROOT)
    CFLA_OUTPUT_ROOT     new-artifact output root   (default: <repo>/outputs)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_ROOT = Path(os.environ.get(
    'CFLA_ARTIFACT_ROOT', REPO_ROOT / 'artifacts')).resolve()

OUTPUT_ROOT = Path(os.environ.get(
    'CFLA_OUTPUT_ROOT', REPO_ROOT / 'outputs')).resolve()

# Optional fallback root consulted only when a file is missing from
# ARTIFACT_ROOT (upstream-layout compatibility).  Read-only.  Defaults to
# ARTIFACT_ROOT because the release ships no separate upstream checkout.
UPSTREAM_ROOT = Path(os.environ.get(
    'CFLA_UPSTREAM_ROOT', ARTIFACT_ROOT)).resolve()


# convenience accessors -------------------------------------------------------
def beam_dir(dataset, affinity='nonneg_pearson_v3'):
    return ARTIFACT_ROOT / f'model/beam_dumps_{dataset}_{affinity}'


def sid_array(dataset, affinity, k):
    return (ARTIFACT_ROOT / 'data' / dataset /
            f'{dataset}_t5_spectral_{affinity}_k{k}_sid{k+1}.npy')


def checkpoint(dataset, affinity, k):
    return (ARTIFACT_ROOT / 'model' / 'ckpt' /
            f'tiger_{dataset}_{affinity}_k{k}_sid{k+1}.pth')


def item_embedding(dataset):
    """Item semantic embeddings (the SID generation INPUT)."""
    return ARTIFACT_ROOT / 'data' / dataset / 'item_emb.parquet'


def item_mapping(dataset):
    """Resolve `data/<dataset>/item_mapping.npy`.

    Resolution order (never a silent fallback - the caller is told which one
    was used via the returned dict):

      1. ARTIFACT_ROOT/data/<dataset>/item_mapping.npy
      2. UPSTREAM_ROOT/data/<dataset>/item_mapping.npy

    Returns a dict with `path`, `source` and `sha256` so provenance can be
    recorded in metadata.  Raises FileNotFoundError if neither exists.
    """
    import hashlib
    cands = [
        (ARTIFACT_ROOT / 'data' / dataset / 'item_mapping.npy', 'artifact_root'),
        (UPSTREAM_ROOT / 'data' / dataset / 'item_mapping.npy', 'upstream_root'),
    ]
    for path, source in cands:
        if path.exists():
            h = hashlib.sha256()
            with path.open('rb') as f:
                for c in iter(lambda: f.read(1 << 20), b''):
                    h.update(c)
            return {'path': str(path), 'source': source,
                    'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    raise FileNotFoundError(
        f"item_mapping.npy for {dataset!r} not found in ARTIFACT_ROOT "
        f"({ARTIFACT_ROOT}) or UPSTREAM_ROOT ({UPSTREAM_ROOT})")


def table_out(name):
    d = OUTPUT_ROOT / 'results' / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def figure_out(name):
    d = OUTPUT_ROOT / 'results' / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def require_artifacts():
    """Fail loudly if the input artifact root is missing or incomplete."""
    if not ARTIFACT_ROOT.exists():
        raise FileNotFoundError(
            f"ARTIFACT_ROOT does not exist: {ARTIFACT_ROOT}\n"
            f"Set CFLA_ARTIFACT_ROOT to the directory holding data/ and "
            f"model/ (see docs/DATA_FORMAT.md).")
    return ARTIFACT_ROOT
