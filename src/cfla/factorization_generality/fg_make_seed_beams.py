#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the missing seed beam dumps (INFERENCE ONLY) for the S-diversity control.

Steps
  1. VERIFY: re-dump Beauty K=2 spectral canonical and require element-wise
     identity with the historical dump.  If this fails, the new dumps are not
     comparable and the driver stops without writing anything.
  2. Dump the missing (dataset, K, seed) cells into
     `outputs/mechanism/factorization_generality/seed_beams/`.

No training is performed: every checkpoint was produced earlier by the
project's own seed-ensemble run.  This script only forward-generates beams.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import (  # noqa: E402
    OUT, SEED_CANONICAL, seed_beam_path, seed_ckpt_path, sid_array_path,
)
from fg_dump_seed_beams import dump, verify_canonical  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402

# Every seed checkpoint that exists, restricted to what is not already dumped.
TARGETS = [
    # (dataset, K, seed)
    ('Beauty', 2, 2028),
    ('Cell_Phones_and_Accessories', 3, 2026),
    ('Cell_Phones_and_Accessories', 3, 2027),
    ('Toys_and_Games', 3, 2026),
]


def main():
    t0 = time.time()
    print('=== step 1: verify dumper reproduces the historical dump ===', flush=True)
    if not verify_canonical():
        print('ABORT: dumper output does not match the historical convention.')
        return 1
    print('verify OK', flush=True)

    log = []
    for ds, k, seed in TARGETS:
        for split in ('test', 'valid'):
            out = OUT / 'seed_beams' / f'{ds}_seed{seed}_{split}_k{k}_sid{k+1}.npz'
            if out.exists():
                print(f'skip existing {out.name}', flush=True)
                log.append((ds, k, seed, split, 'exists'))
                continue
            ck = seed_ckpt_path(ds, k, seed)
            if not ck.exists():
                print(f'MISSING ckpt {ck}')
                log.append((ds, k, seed, split, 'no_ckpt'))
                continue
            code = sid_array_path(ds, 'spectral', k)
            print(f'=== dump {ds} K={k} seed={seed} split={split} ===', flush=True)
            dump(str(rp.ARTIFACT_ROOT / 'data' / ds), str(code), str(ck), str(out),
                 split, k + 1, ds)
            log.append((ds, k, seed, split, 'dumped'))
    print(f'=== done in {time.time()-t0:.0f}s ===')
    for row in log:
        print(row)
    # inventory of every seed beam now reachable
    print('--- reachable seed beams ---')
    for ds, k, seed in [('Beauty', 2, 2026), ('Beauty', 2, 2027), ('Beauty', 2, 2028),
                        ('Beauty', 3, 2026), ('Beauty', 4, 2026), ('Beauty', 5, 2026),
                        ('Cell_Phones_and_Accessories', 3, 2026),
                        ('Cell_Phones_and_Accessories', 3, 2027),
                        ('Toys_and_Games', 3, 2026)]:
        p = seed_beam_path(ds, k, seed, 'test')
        print(f'  {ds} k={k} seed={seed}: {p.exists()} {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
