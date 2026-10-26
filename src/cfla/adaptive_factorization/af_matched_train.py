#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP A of the adaptive-factorization study.

COMPUTE-MATCHED SHARED-FACTORIZATION CONTROL
--------------------------------------------

The previous shared model (`xt_train_shared.py --exposure balanced`) showed every original
training instance once under F1 *and* once under F2 every epoch, i.e. it executed
~2x the optimizer steps of a single independent model.  Its reported gains over the
independent models are therefore confounded with compute.

This driver keeps the architecture, optimizer, learning rate, batch size, patience,
beam size, seed and validation-selection rule **identical** to the previous shared run
and changes exactly one thing: the factorization exposure schedule.

    every original training instance is shown exactly ONCE per epoch,
    under exactly ONE factorization, F1 or F2.

Assignment is DETERMINISTIC and BALANCED:

  * the epoch's assignment is a permutation of a multiset holding
    `ceil(n/2)` copies of F1 and `floor(n/2)` copies of F2, so
    `|count(F1) - count(F2)| <= 1` exactly;
  * the permutation is drawn from `numpy.random.default_rng(epoch_seed)` where
    `epoch_seed = seed * 1000003 + epoch` (documented, reproducible, and
    independent of Python/numpy global state);
  * because both factorizations' training sets are index-aligned (same parquet row
    order, same sliding-window expansion), index `i` denotes the same original
    instance under either factorization.

Consequences that are reported, not hidden:

  * training samples per epoch == the canonical single-model sample count
    (Beauty: 131410, vs 262820 previously);
  * optimizer steps per epoch ~= 514 (Beauty), vs 1027 previously;
  * a batch now mixes both control tokens, since the assignment is random rather
    than blocked by factorization.  This is part of the exposure schedule.

Usage
  python3 af_matched_train.py --dataset Beauty \
      --outdir outputs/adaptive_factorization/compute_matched \
      --save <ckpt.pth> --log <train.log> --run_dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xt_common import CODEBOOK, STEP2, TOKENIZERS, sha256_file, shared_vocab_plan, sid_path
from xt_train_shared import (CANONICAL, ControlPrefixedDataset, build_factorization_data,
                             build_cfg)
from src.tiger.dataloader import GenRecDataLoader
from src.tiger.train import TIGER, evaluate_item_level, set_seed

SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# deterministic balanced assignment
# ---------------------------------------------------------------------------
def epoch_seed(base_seed: int, epoch: int, stride: int = SEED_STRIDE) -> int:
    """Assignment seed for `epoch` (0-based).  Deterministic and reproducible."""
    if epoch < 0:
        raise ValueError('epoch must be >= 0')
    return int(base_seed) * int(stride) + int(epoch)


def balanced_assignment(n: int, base_seed: int, epoch: int, n_arms: int = 2):
    """Return (assign, seed) with balanced arms and |count difference| <= 1.

    `assign[i]` is the arm index used for original instance `i` in `epoch`.
    """
    if n_arms < 2:
        raise ValueError('n_arms must be >= 2')
    if n <= 0:
        raise ValueError('n must be positive')
    seed = epoch_seed(base_seed, epoch)
    rng = np.random.default_rng(seed)
    base = n // n_arms
    counts = np.full(n_arms, base, dtype=np.int64)
    # distribute the remainder over the first arms (<= 1 extra each)
    counts[: n - base * n_arms] += 1
    multiset = np.repeat(np.arange(n_arms, dtype=np.int64), counts)
    if multiset.size != n:
        raise AssertionError('balanced assignment produced the wrong size')
    rng.shuffle(multiset)
    return multiset, seed


class BalancedMatchedDataset:
    """One factorization per original instance per epoch, deterministically balanced.

    Both wrapped per-factorization datasets must be index-aligned: entry `i` of each is
    the same (user, history, target) instance written in a different factorization.
    """

    def __init__(self, per_f, base_seed, n_arms=None, stride=SEED_STRIDE):
        self.keys = list(per_f)
        self.per_f = {k: per_f[k] for k in self.keys}
        self.n_arms = len(self.keys) if n_arms is None else int(n_arms)
        if self.n_arms != len(self.keys):
            raise ValueError('n_arms must equal the number of factorizations')
        lens = {k: len(self.per_f[k]) for k in self.keys}
        if len(set(lens.values())) != 1:
            raise ValueError(f'factorization train sets are not index-aligned in size: {lens}')
        self.n = lens[self.keys[0]]
        self.base_seed = int(base_seed)
        self.stride = int(stride)
        self.epoch = -1
        self.seed = None
        self.assign = None
        self.counts = None
        self.new_epoch()

    def new_epoch(self):
        """Advance to the next epoch and redraw the assignment."""
        self.epoch += 1
        self.assign, self.seed = balanced_assignment(
            self.n, self.base_seed, self.epoch, self.n_arms)
        self.counts = {self.keys[a]: int((self.assign == a).sum())
                       for a in range(self.n_arms)}
        return self.assign

    def epoch_record(self):
        return {'epoch_index0': int(self.epoch), 'assignment_seed': int(self.seed),
                'assignment_counts': self.counts,
                'assignment_balance_max_abs_diff': int(
                    max(self.counts.values()) - min(self.counts.values()))}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.per_f[self.keys[int(self.assign[i])]][i]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--factorizations', default='spectral_k3:rqvae3')
    ap.add_argument('--save', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--seed', type=int, default=CANONICAL['seed'])
    ap.add_argument('--num_epochs', type=int, default=CANONICAL['num_epochs'])
    ap.add_argument('--batch_size', type=int, default=CANONICAL['batch_size'])
    ap.add_argument('--infer_size', type=int, default=CANONICAL['infer_size'])
    ap.add_argument('--lr', type=float, default=CANONICAL['lr'])
    ap.add_argument('--early_stop', type=int, default=CANONICAL['early_stop'])
    ap.add_argument('--beam_size', type=int, default=CANONICAL['beam_size'])
    ap.add_argument('--max_len', type=int, default=CANONICAL['max_len'])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    ds = a.dataset
    factorizations = a.factorizations.split(':')
    log_path = Path(a.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(log_path), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Configuration: {vars(a)}')

    set_seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'device={dev}')

    train_data, plan = build_factorization_data(ds, factorizations, 'train', a.max_len, CODEBOOK)
    valid_data, _ = build_factorization_data(ds, factorizations, 'valid', a.max_len, CODEBOOK)

    cfg = build_cfg(plan['vocab_size'])
    model = TIGER(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[matched] params={n_params} vocab={plan["vocab_size"]} ctrl={plan["control_tokens"]}',
          flush=True)
    logging.info(f'params={n_params} vocab={plan["vocab_size"]} ctrl={plan["control_tokens"]}')

    train_ds = BalancedMatchedDataset({f: train_data[f] for f in factorizations},
                                      base_seed=a.seed)
    train_loader = GenRecDataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    valid_loaders = {f: GenRecDataLoader(valid_data[f], batch_size=a.infer_size, shuffle=False)
                     for f in factorizations}

    optimizer = optim.Adam(model.parameters(), lr=a.lr)
    model.to(dev)

    log = {
        'dataset': ds, 'factorizations': factorizations, 'plan': plan,
        'exposure': 'compute_matched_balanced',
        'assignment': {
            'scheme': 'deterministic balanced permutation',
            'rng': 'numpy.random.default_rng',
            'seed_formula': f'seed * {SEED_STRIDE} + epoch_index0',
            'base_seed': a.seed, 'stride': SEED_STRIDE,
            'balance_guarantee': '|count(F1) - count(F2)| <= 1',
            'index_alignment': 'entry i is the same original instance in every factorization',
        },
        'params': n_params, 'vocab_size': plan['vocab_size'],
        'config': {**CANONICAL, 'seed': a.seed},
        'train_size_per_factorization': {f: len(train_data[f]) for f in factorizations},
        'train_size_total': len(train_ds),
        'canonical_single_model_train_size': len(train_data[factorizations[0]]),
        'epochs': [],
    }
    Path(a.run_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.run_dir) / 'compute_matched_config.json').write_text(json.dumps(log, indent=2))

    best_macro = -1.0
    patience = 0
    total_steps = 0
    t0 = time.time()
    for epoch in range(a.num_epochs):
        if epoch > 0:
            train_ds.new_epoch()
        rec = train_ds.epoch_record()
        ep_t0 = time.time()
        model.train()
        total, steps = 0.0, 0
        for batch in train_loader:
            input_ids = batch['history'].to(dev)
            am = batch['attention_mask'].to(dev)
            labels = batch['target'].to(dev)
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=am, labels=labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        train_loss = total / max(steps, 1)
        total_steps += steps
        tr_s = time.time() - ep_t0

        per_f, val_t0 = {}, time.time()
        for f in factorizations:
            _, nd = evaluate_item_level(
                model, valid_loaders[f], CANONICAL['topk_list'], a.beam_size, dev,
                str(sid_path(f, ds)), CODEBOOK)
            per_f[f] = float(nd['NDCG@20'])
        val_s = time.time() - val_t0
        macro = float(np.mean(list(per_f.values())))

        ep = {'epoch': epoch + 1, **rec, 'train_loss': train_loss,
              'train_seconds': round(tr_s, 1), 'valid_seconds': round(val_s, 1),
              'steps': steps, 'total_steps': total_steps,
              'valid_ndcg20': per_f, 'macro_valid_ndcg20': macro,
              'seconds': round(time.time() - ep_t0, 1)}
        log['epochs'].append(ep)
        logging.info(f"Epoch {epoch+1}/{a.num_epochs} loss={train_loss:.6f} macro={macro:.6f} "
                     f"counts={rec['assignment_counts']} seed={rec['assignment_seed']}")
        print(f"[matched {ds}] epoch {epoch+1} loss={train_loss:.5f} macro={macro:.6f} "
              f"counts={rec['assignment_counts']} "
              f"{ {k: round(v,5) for k,v in per_f.items()} } ({ep['seconds']}s)", flush=True)

        improved = macro > best_macro
        if improved:
            best_macro = macro
            patience = 0
            Path(a.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.save)
            log['best_epoch'] = epoch + 1
            log['best_macro_valid_ndcg20'] = macro
            log['best_epoch_record'] = ep
            log['checkpoint_sha256'] = sha256_file(a.save)
            logging.info(f'Best macro {macro} saved to {a.save}')
        else:
            patience += 1
            logging.info(f'No improvement. Early stop counter: {patience}')
        log['elapsed_s'] = round(time.time() - t0, 1)
        log['total_optimizer_steps'] = total_steps
        (Path(a.run_dir) / 'train_log.json').write_text(json.dumps(log, indent=2))
        if not improved and patience >= a.early_stop:
            logging.info('Early stopping triggered.')
            break

    log['elapsed_s'] = round(time.time() - t0, 1)
    log['total_optimizer_steps'] = total_steps
    log['best_macro_valid_ndcg20'] = best_macro
    log['checkpoint_sha256'] = sha256_file(a.save)
    (Path(a.run_dir) / 'train_log.json').write_text(json.dumps(log, indent=2))
    print(f'[matched {ds}] done best_macro={best_macro:.6f} steps={total_steps} '
          f'elapsed={log["elapsed_s"]}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
