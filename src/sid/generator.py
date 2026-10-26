#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified SID generation CLI.

    python -m src.sid.generator --dataset Beauty --affinity nonneg_pearson \
        --k 2 --codebook_size 256 --seed 2025

    python -m src.sid.generator --config configs/experiments/<name>.yaml

Pipeline (the audited canonical spectral SID construction):

    item semantic embedding
      -> dimension affinity matrix          (affinity.py)
      -> spectral clustering of dimensions  (spectral_partition.py)
      -> K semantic subspaces
      -> subspace-wise k-means quantization (quantizer.py)
      -> base semantic SID
      -> collision detection + resolution   (collision.py)
      -> final SID array  (L = K + 1)
      -> item <-> SID mapping + artifacts   (io.py)

Writes ONLY to OUTPUT_ROOT/outputs/sids/... ; writing into ARTIFACT_ROOT is
refused. No source patching, no hard-coded dataset or checkpoint: everything
comes from CLI arguments or a YAML config.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ''):                       # allow `python src/sid/generator.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sid.affinity import build_dimension_affinity, CANONICAL_AFFINITY  # noqa: E402
from src.sid.collision import add_collision_token, collision_stats, compute_collision_rate  # noqa: E402
from src.sid.embedding import (  # noqa: E402
    ALIGNMENT_MODES, align_embeddings, load_item_embedding,
)
from src.sid.io import GENERATOR_VERSION, sid_dir, write_sid_artifacts  # noqa: E402
from src.sid.quantizer import generate_base_sid_by_group_kmeans  # noqa: E402
from src.sid.spectral_partition import partition_sizes, spectral_group_dimensions  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402
from src.utils.run_metadata import RunMetadata  # noqa: E402

DEFAULTS = {
    'dataset': None,
    'affinity': CANONICAL_AFFINITY,
    'k': None,
    'codebook_size': 256,
    'seed': 2025,
    'pca_dim': 64,
    'item_emb_path': None,
    'alignment_mode': 'canonical',
}


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m src.sid.generator',
        description='Generate a spectral Semantic ID mapping.')
    p.add_argument('--config', type=str, default=None,
                   help='YAML config; CLI flags override its values.')
    p.add_argument('--dataset', type=str, default=None)
    p.add_argument('--affinity', type=str, default=None,
                   help='nonneg_pearson (canonical) | abs_pearson | cosine')
    p.add_argument('--k', type=int, default=None, help='number of semantic subspaces')
    p.add_argument('--codebook_size', type=int, default=None)
    p.add_argument('--seed', type=int, default=None, help='random_state')
    p.add_argument('--pca_dim', type=int, default=None,
                   help='PCA-whitening target inside each subspace; 0 disables')
    p.add_argument('--item_emb_path', type=str, default=None,
                   help='override the embedding parquet path')
    p.add_argument('--alignment-mode', dest='alignment_mode', type=str,
                   default=None, choices=list(ALIGNMENT_MODES),
                   help="'canonical' (default) uses the parquet row order and "
                        "treats row i as item i+1, byte-identical to the audited "
                        "runs. 'item_id' reorders rows by the embedding file's "
                        "own item-id column so row i really is item i+1.")
    p.add_argument('--print-metadata', action='store_true',
                   help='print the metadata JSON to stdout at the end')
    return p


def resolve_config(args) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        # accept both a flat config and one nested under `sid:`
        cfg.update(loaded.get('sid', loaded))
    for key in ('dataset', 'affinity', 'k', 'codebook_size', 'seed', 'pca_dim',
                'item_emb_path', 'alignment_mode'):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if not cfg['dataset']:
        raise SystemExit('error: --dataset (or a config providing it) is required')
    if not cfg['k']:
        raise SystemExit('error: --k (or a config providing it) is required')
    return cfg


def generate(cfg: dict, verbose: bool = True) -> dict:
    dataset = cfg['dataset']
    affinity = cfg['affinity']
    k = int(cfg['k'])
    codebook_size = int(cfg['codebook_size'])
    seed = int(cfg['seed'])
    pca_dim = int(cfg['pca_dim'])
    alignment_mode = cfg.get('alignment_mode', 'canonical')
    sid_len = k + 1

    emb_path = cfg.get('item_emb_path') or rp.item_embedding(dataset)
    emb_path = Path(emb_path)

    run = RunMetadata(stage='sid_generation', config=cfg, inputs=[emb_path])

    X, df = load_item_embedding(emb_path)
    X, align_meta = align_embeddings(X, df, alignment_mode)
    if verbose:
        print(f"[sid] embeddings     {X.shape} from {emb_path}")
        print(f"[sid] alignment      {alignment_mode}  "
              f"rows_reordered={align_meta['number_rows_reordered']} "
              f"({align_meta['percentage_rows_reordered']}%)")

    # 1-3. affinity -> spectral partition -> K subspaces
    # (affinity is built inside spectral_group_dimensions; also computed here so
    #  it can be asserted and recorded)
    groups, dim_labels = spectral_group_dimensions(
        X, num_groups=k, metric=affinity, random_state=seed)
    sizes = partition_sizes(groups)
    if verbose:
        print(f"[sid] K={k} group sizes {sizes}")

    S = build_dimension_affinity(X, metric=affinity, )
    assert S.shape == (X.shape[1], X.shape[1])
    assert np.isfinite(S).all(), 'affinity has non-finite entries'
    assert np.allclose(S, S.T, atol=1e-5), 'affinity is not symmetric'
    assert S.min() >= 0.0, 'affinity has negative entries'

    # 4-5. subspace quantization -> base SID
    base = generate_base_sid_by_group_kmeans(
        X, groups, codebook_size=codebook_size, pca_dim=pca_dim,
        random_state=seed)

    # 6-7. collision detection + resolution token -> final SID
    cstats = collision_stats(base)
    final = add_collision_token(base, codebook_size=codebook_size)
    _, n_unique, n_total = compute_collision_rate(final)

    # correctness assertions (design guarantees, not performance claims)
    assert final.shape == (X.shape[0], sid_len), (final.shape, sid_len)
    assert base.shape == (X.shape[0], k)
    assert final[:, :k].tolist() == base.tolist(), 'semantic prefix was modified'
    assert n_unique == n_total, 'final SID is not injective'
    assert final[:, :-1].min() >= 0 and final[:, :-1].max() < codebook_size

    metadata = {
        'dataset': dataset,
        'affinity': affinity,
        'K': k,
        'sid_length': sid_len,
        'codebook_size': codebook_size,
        'seed': seed,
        'pca_dim': pca_dim,
        'embedding_source': str(emb_path),
        'embedding_shape': list(X.shape),
        'git_commit': run.data.get('git_commit'),
        'timestamp': run.data['start_timestamp'],
        'generator_version': GENERATOR_VERSION,
        'alignment': align_meta,
        'num_items': int(X.shape[0]),
        'embedding_dim': int(X.shape[1]),
        'group_sizes': sizes,
        'prefix_collision_rate': cstats['prefix_collision_rate'],
        'prefix_uniqueness': cstats['prefix_uniqueness'],
        'number_collision_groups': cstats['num_collision_groups'],
        'max_collision_group': cstats['max_collision_group'],
        'final_unique_sid_count': int(n_unique),
        'final_sid_injective': bool(n_unique == n_total),
        'item_mapping': _try_item_mapping(dataset),
    }

    out_dir = write_sid_artifacts(dataset, affinity, k, seed, final, base,
                                  groups, dim_labels, metadata)
    meta_path = run.finish(outputs=[out_dir],
                           extra={'sid_dir': str(out_dir),
                                  'alignment': align_meta})
    if verbose:
        print(f"[sid] prefix collision {cstats['prefix_collision_rate']:.4f}  "
              f"max group {cstats['max_collision_group']}  "
              f"final unique {n_unique}/{n_total}  (injective: "
              f"{n_unique == n_total})")
        print(f"[sid] wrote {out_dir}")
        print(f"[sid] run metadata {meta_path}")
    return {'out_dir': out_dir, 'metadata': metadata, 'final': final,
            'base': base, 'groups': groups, 'dim_labels': dim_labels}


def _try_item_mapping(dataset):
    try:
        return rp.item_mapping(dataset)
    except FileNotFoundError as e:
        return {'missing': True, 'reason': str(e)}


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    res = generate(cfg)
    if args.print_metadata:
        print(json.dumps(res['metadata'], indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
