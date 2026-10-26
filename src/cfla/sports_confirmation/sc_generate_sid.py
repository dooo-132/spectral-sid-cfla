#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP A — generate the Sports_and_Outdoors Spectral-K3 SID with the audited pipeline.

The algorithm is the repository's own certified generator, imported and called verbatim:

    src.sid.embedding.load_item_embedding / align_embeddings
    src.sid.spectral_partition.spectral_group_dimensions
    src.sid.quantizer.generate_base_sid_by_group_kmeans
    src.sid.collision.add_collision_token

Only the *write target* differs from `src.sid.generator.generate()`: that function writes
to `outputs/sids/<dataset>/...`, and this round is permitted to write only under
`outputs/sports_confirmation/`, so the same arrays are written here in the same layout
(sid.npy, base_sid.npy, dim_groups.npy, dim_labels.npy, metadata.json, manifest.sha256).

Certification: before this script was used on Sports, the identical call sequence was run
on Beauty and reproduced `Beauty_t5_spectral_nonneg_pearson_v3_k3_sid4.npy` **byte for
byte** (base_sid and final SID both element-wise identical). That check is re-run by
`sc_sid_audit.py`.

Canonical parameters (from src.sid.generator.DEFAULTS):
    affinity=nonneg_pearson, K=3, codebook_size=256, pca_dim=64, seed=2025,
    alignment_mode='canonical'

Usage
  python3 sc_generate_sid.py --dataset Sports_and_Outdoors --outdir outputs/sports_confirmation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.sid.collision import add_collision_token, collision_stats  # noqa: E402
from src.sid.embedding import align_embeddings, load_item_embedding  # noqa: E402
from src.sid.generator import DEFAULTS  # noqa: E402
from src.sid.quantizer import generate_base_sid_by_group_kmeans  # noqa: E402
from src.sid.spectral_partition import spectral_group_dimensions  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def generate_sid(dataset, out_dir, verbose=True):
    cfg = dict(DEFAULTS)
    cfg['dataset'] = dataset
    cfg['k'] = 3
    emb_path = Path(rp.item_embedding(dataset))

    t0 = time.time()
    X, df = load_item_embedding(emb_path)
    X, align = align_embeddings(X, df, cfg['alignment_mode'])
    if verbose:
        print(f'[sid] embeddings {X.shape} from {emb_path}')
        print(f'[sid] alignment {cfg["alignment_mode"]} '
              f'rows_reordered={align["number_rows_reordered"]}')

    groups, dim_labels = spectral_group_dimensions(
        X, num_groups=cfg['k'], metric=cfg['affinity'], random_state=cfg['seed'])
    if verbose:
        print(f'[sid] K={cfg["k"]} group sizes {[len(g) for g in groups]}')

    base = generate_base_sid_by_group_kmeans(
        X, groups, codebook_size=cfg['codebook_size'], pca_dim=cfg['pca_dim'],
        random_state=cfg['seed'])
    cstats = collision_stats(base)
    final = add_collision_token(base, codebook_size=cfg['codebook_size'])

    sid_len = cfg['k'] + 1
    assert final.shape == (X.shape[0], sid_len), (final.shape, sid_len)
    assert base.shape == (X.shape[0], cfg['k'])
    assert final[:, :cfg['k']].tolist() == base.tolist(), 'semantic prefix modified'
    assert len({tuple(r) for r in final}) == len(final), 'final SID not injective'
    assert final[:, :-1].min() >= 0 and final[:, :-1].max() < cfg['codebook_size']

    d = Path(out_dir) / 'sid' / 'spectral_k3'
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f'HARD FAIL: {d} already exists and is non-empty')
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'sid.npy', final)
    np.save(d / 'base_sid.npy', base)
    np.save(d / 'dim_groups.npy', np.array(list(groups), dtype=object), allow_pickle=True)
    np.save(d / 'dim_labels.npy', dim_labels)

    meta = {
        'dataset': dataset, 'affinity': cfg['affinity'], 'K': cfg['k'],
        'sid_length': sid_len, 'codebook_size': cfg['codebook_size'],
        'seed': cfg['seed'], 'pca_dim': cfg['pca_dim'],
        'alignment_mode': cfg['alignment_mode'],
        'embedding_source': str(emb_path),
        'embedding_sha256': sha256_file(emb_path),
        'embedding_shape': list(X.shape),
        'num_items': int(X.shape[0]), 'embedding_dim': int(X.shape[1]),
        'group_sizes': [int(len(g)) for g in groups],
        'prefix_collision_rate': cstats['prefix_collision_rate'],
        'prefix_uniqueness': cstats['prefix_uniqueness'],
        'number_collision_groups': cstats['num_collision_groups'],
        'max_collision_group': cstats['max_collision_group'],
        'final_unique_sid_count': int(len({tuple(r) for r in final})),
        'final_sid_injective': True,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'generator': 'src.sid.generator functions, called verbatim',
        'write_target_note': ('written under outputs/sports_confirmation/ instead of '
                              'outputs/sids/ to honour this round\'s write restriction; '
                              'the array-producing code is unchanged'),
    }
    (d / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    files = ['sid.npy', 'base_sid.npy', 'dim_groups.npy', 'dim_labels.npy', 'metadata.json']
    (d / 'manifest.sha256').write_text(
        '\n'.join(f'{sha256_file(d/f)}  {f}' for f in files) + '\n')
    if verbose:
        print(f'[sid] prefix collision {cstats["prefix_collision_rate"]:.4f}  '
              f'max group {cstats["max_collision_group"]}  '
              f'injective True  ({time.time()-t0:.1f}s)')
        print(f'[sid] wrote {d}')
    return {'out_dir': d, 'metadata': meta, 'final': final}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Sports_and_Outdoors')
    ap.add_argument('--outdir', default='outputs/sports_confirmation')
    a = ap.parse_args(argv)
    r = generate_sid(a.dataset, a.outdir)
    print(json.dumps({k: v for k, v in r['metadata'].items()
                      if k in ('num_items', 'sid_length', 'codebook_size',
                               'prefix_collision_rate', 'max_collision_group')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
