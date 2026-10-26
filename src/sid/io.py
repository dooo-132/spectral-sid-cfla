#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SID artifact serialization and loading.

OUTPUT LAYOUT (this clean repo only -- never write to ARTIFACT_ROOT):

    outputs/sids/<dataset>/<affinity>/k<K>/seed<S>/
        sid.npy           (N, K+1) final SIDs
        base_sid.npy      (N, K)   semantic prefix (may collide)
        dim_groups.npy    (K,) object array of dimension-index arrays
        dim_labels.npy    (d,)  group index of every embedding dimension
        metadata.json     full provenance + statistics
        manifest.sha256   hashes of every file in this directory

Everything is written under OUTPUT_ROOT by construction; `write_sid_artifacts`
refuses a destination inside ARTIFACT_ROOT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 'sidgen-1.0.0'


def assert_not_artifact_root(path: Path) -> Path:
    """Hard guard: refuse to write anywhere inside the read-only evidence repo."""
    from ..utils import repo_paths as rp
    p = Path(path).resolve()
    root = rp.ARTIFACT_ROOT.resolve()
    if p == root or root in p.parents:
        raise PermissionError(
            f"refusing to write inside ARTIFACT_ROOT ({root}): {p}\n"
            f"New artifacts must go to OUTPUT_ROOT ({rp.OUTPUT_ROOT}).")
    return p


def sid_dir(dataset: str, affinity: str, k: int, seed: int):
    from ..utils import repo_paths as rp
    return (rp.OUTPUT_ROOT / 'sids' / dataset / affinity /
            f'k{k}' / f'seed{seed}')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write_sid_artifacts(dataset: str, affinity: str, k: int, seed: int,
                        final_sid: np.ndarray, base_sid: np.ndarray,
                        groups, dim_labels: np.ndarray, metadata: dict) -> Path:
    """Write all SID artifacts plus metadata.json and manifest.sha256."""
    d = assert_not_artifact_root(sid_dir(dataset, affinity, k, seed))
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / 'sid.npy', final_sid)
    np.save(d / 'base_sid.npy', base_sid)
    np.save(d / 'dim_groups.npy', np.array(list(groups), dtype=object),
            allow_pickle=True)
    np.save(d / 'dim_labels.npy', dim_labels)

    meta = dict(metadata)
    meta['generator_version'] = GENERATOR_VERSION
    (d / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))

    lines = []
    for name in ('sid.npy', 'base_sid.npy', 'dim_groups.npy', 'dim_labels.npy',
                 'metadata.json'):
        lines.append(f"{sha256_file(d / name)}  {name}")
    (d / 'manifest.sha256').write_text('\n'.join(lines) + '\n')
    return d


def load_sid_dir(d: Path):
    """Load a SID directory written by `write_sid_artifacts`."""
    d = Path(d)
    out = {
        'sid': np.load(d / 'sid.npy'),
        'base_sid': np.load(d / 'base_sid.npy'),
        'dim_labels': np.load(d / 'dim_labels.npy'),
        'groups': list(np.load(d / 'dim_groups.npy', allow_pickle=True)),
    }
    meta = d / 'metadata.json'
    out['metadata'] = json.loads(meta.read_text()) if meta.exists() else {}
    return out
