#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the four primary Shared-T5 runs.

Runs the 4 datasets two at a time (the model is ~4.6M parameters, so two
concurrent beam-search trainings fit comfortably and roughly halve wall-clock).
Per-dataset configs are written to `step2_shared_model/configs/` before launch.

Primary factorization set: Spectral-K3 + RQ-VAE-3layer (STEP 1 fairness audit PASS).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, STEP2  # noqa: E402

PY = sys.executable
TRAINER = str(HERE / 'xt_train_shared.py')
FACTORIZATIONS = 'spectral_k3:rqvae3'
CONFIGS = STEP2 / 'configs'
RUNS = STEP2 / 'runs'
LOGS = STEP2 / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


def launch(ds):
    run_dir = RUNS / f'{ds}_shared_spec_rqvae_seed2025'
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        'dataset': ds, 'factorizations': FACTORIZATIONS, 'exposure': 'balanced',
        'seed': 2025, 'num_epochs': 200, 'batch_size': 256, 'infer_size': 96,
        'lr': 1e-4, 'early_stop': 10, 'beam_size': 30, 'max_len': 20,
        'canonical_capacity': {'num_layers': 4, 'num_decoder_layers': 4,
                               'd_model': 128, 'd_ff': 1024, 'num_heads': 6,
                               'd_kv': 64, 'dropout_rate': 0.1},
        'selection': 'macro-average item-level validation NDCG@20 across factorizations',
        'test_used_for_selection': False,
    }
    (CONFIGS / f'{ds}_shared.yaml').write_text(json.dumps(cfg, indent=2))
    cmd = [PY, TRAINER, '--dataset', ds, '--factorizations', FACTORIZATIONS,
           '--save', str(run_dir / 'shared_model.pth'),
           '--log', str(LOGS / f'{ds}_shared.log'),
           '--run_dir', str(run_dir)]
    env = dict(os.environ, TQDM_DISABLE='1')
    f = open(LOGS / f'{ds}_shared.stdout', 'w')
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f'launched {ds} pid={p.pid}')
    return p, f


def main():
    pairs = [('Beauty', 'Toys_and_Games'),
             ('Cell_Phones_and_Accessories', 'Health_and_Personal_Care')]
    for pair in pairs:
        procs = []
        for ds in pair:
            procs.append(launch(ds))
        for p, f in procs:
            p.wait()
            f.close()
        print(f'pair done: {pair} at {time.strftime("%H:%M:%S")}', flush=True)
    print('all shared models done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
