#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — Factorization-Controlled Shared T5.

ONE encoder, ONE decoder, ONE embedding table, ONE output head.  The only
factorization-specific information the model receives is

  (a) a reserved control token `<F_x>`, prepended to the encoder input, and
  (b) which factorization's SID sequence the history and the target are written
      in,

so the same (user, history, target) interaction becomes a different discrete
sequence under each factorization.

Every original training instance is presented exactly once per factorization per
epoch (factorization-balanced exposure).  Validation early stopping uses the
MACRO AVERAGE of item-level validation NDCG@20 across factorizations, each
factorization evaluated with its own control token and its own SID array.  Test
is never used for selection.

Evaluation only reuses the canonical `src.tiger.train.evaluate_item_level`
unchanged.

Usage
  python3 xt_train_shared.py --dataset Beauty \
      --factorizations spectral_k3:rqvae3 --outdir ... --save <ckpt.pth>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import (ART, CODEBOOK, DATASETS, SHORT, STEP2, TOKENIZERS,
                       ckpt_path, load_sid, sha256_file, shared_vocab_plan,
                       sid_path)  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402
from src.tiger.train import TIGER, evaluate_item_level, set_seed  # noqa: E402

CANONICAL = {
    'batch_size': 256, 'infer_size': 96, 'num_epochs': 200, 'lr': 1e-4,
    'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
    'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
    'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
    'max_len': 20, 'seed': 2025, 'early_stop': 10, 'beam_size': 30,
    'topk_list': [5, 10, 20],
}


def build_cfg(vocab_size):
    """The single canonical shared-model config, parameterised by vocabulary size."""
    return {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
            'vocab_size': int(vocab_size), 'pad_token_id': 0, 'eos_token_id': 0,
            'feed_forward_proj': 'relu'}


class ControlPrefixedDataset(Dataset):
    """Wrap a canonical GenRecDataset, prepending the factorization control token.

    The control token is a single-element list at the front of the history token
    list, so the flattened encoder input is `[ctrl] + history tokens`.
    """

    def __init__(self, base: GenRecDataset, control_id: int):
        self.ctrl = int(control_id)
        self.data = [{'history': [[self.ctrl]] + list(d['history']),
                      'target': d['target']} for d in base.data]
        base.data = []          # the wrapped copies own the structure from here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class StepMatchedDataset(Dataset):
    """Compute-matched control: each original instance is shown ONCE per epoch, under a
    uniformly random factorization, so optimizer steps stay ~= a single model."""

    def __init__(self, per_f, seed):
        self.per_f = per_f
        self.keys = list(per_f)
        self.n = len(per_f[self.keys[0]])
        for f in self.keys:
            if len(per_f[f]) != self.n:
                raise ValueError('factorization train sets differ in length')
        self.rng = np.random.default_rng(seed)
        self.new_epoch()

    def new_epoch(self):
        self.assign = self.rng.integers(0, len(self.keys), size=self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.per_f[self.keys[int(self.assign[i])]][i]


def build_factorization_data(ds, factorizations, split, max_len, codebook):
    """Return {F: ControlPrefixedDataset} for one split."""
    plan = shared_vocab_plan(factorizations)
    out = {}
    for f in factorizations:
        base = GenRecDataset(
            dataset_path=str(ART / 'data' / ds / f'{split}.parquet'),
            code_path=str(sid_path(f, ds)), mode=('train' if split == 'train' else 'evaluation'),
            max_len=max_len, codebook_size=codebook)
        out[f] = ControlPrefixedDataset(base, plan['control_tokens'][f])
    return out, plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--factorizations', required=True, help='colon-separated keys')
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
    ap.add_argument('--exposure', choices=['balanced', 'step_matched'],
                    default='balanced',
                    help='balanced: every instance once per factorization per epoch '
                         '(M x samples). step_matched: sample ONE factorization per '
                         'instance at random, keeping optimizer steps ~= single model.')
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

    train_data, plan = build_factorization_data(ds, factorizations, 'train',
                                               a.max_len, CODEBOOK)
    valid_data, _ = build_factorization_data(ds, factorizations, 'valid',
                                            a.max_len, CODEBOOK)

    cfg = {k: CANONICAL[k] for k in
           ('num_layers', 'num_decoder_layers', 'd_model', 'd_ff', 'num_heads',
            'd_kv', 'dropout_rate', 'pad_token_id', 'eos_token_id',
            'feed_forward_proj')}
    cfg['vocab_size'] = plan['vocab_size']
    model = TIGER(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f'shared model vocab_size={plan["vocab_size"]} control_tokens='
                 f'{plan["control_tokens"]} params={n_params}')
    print(f'shared model params={n_params} vocab={plan["vocab_size"]} '
          f'control={plan["control_tokens"]}')

    if a.exposure == 'balanced':
        train_ds = ConcatDataset([train_data[f] for f in factorizations])
    else:
        train_ds = StepMatchedDataset({f: train_data[f] for f in factorizations},
                                      seed=a.seed)
    train_loader = GenRecDataLoader(train_ds, batch_size=a.batch_size, shuffle=True)

    valid_loaders = {f: GenRecDataLoader(valid_data[f], batch_size=a.infer_size,
                                         shuffle=False) for f in factorizations}

    optimizer = optim.Adam(model.parameters(), lr=a.lr)
    model.to(dev)

    valid_sizes = {f: len(valid_data[f]) for f in factorizations}
    log = {'dataset': ds, 'factorizations': factorizations, 'plan': plan,
           'params': n_params, 'vocab_size': plan['vocab_size'],
           'config': {**CANONICAL, **{'exposure': a.exposure, 'seed': a.seed}},
           'train_size_per_factorization': {f: len(train_data[f]) for f in factorizations},
           'train_size_total': len(train_ds), 'epochs': []}

    best_macro = -1.0
    patience = 0
    t0 = time.time()
    for epoch in range(a.num_epochs):
        ep_t0 = time.time()
        if a.exposure == 'step_matched':
            train_ds.new_epoch()
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
            total += loss.item()
            steps += 1
        train_loss = total / max(steps, 1)
        tr_s = time.time() - ep_t0

        per_f, val_s = {}, time.time() - ep_t0 - tr_s
        for f in factorizations:
            _, nd = evaluate_item_level(
                model, valid_loaders[f], CANONICAL['topk_list'], a.beam_size, dev,
                str(sid_path(f, ds)), CODEBOOK)
            per_f[f] = float(nd['NDCG@20'])
        macro = float(np.mean(list(per_f.values())))
        ep = {'epoch': epoch + 1, 'train_loss': train_loss,
              'train_seconds': round(tr_s, 1), 'steps': steps,
              'valid_ndcg20': per_f, 'macro_valid_ndcg20': macro,
              'seconds': round(time.time() - ep_t0, 1)}
        log['epochs'].append(ep)
        logging.info(f"Epoch {epoch+1}/{a.num_epochs} loss={train_loss:.6f} "
                     f"valid={ {k: round(v,6) for k,v in per_f.items()} } macro={macro:.6f}")
        print(f"[{ds}] epoch {epoch+1} loss={train_loss:.5f} macro={macro:.6f} "
              f"{ {k: round(v,5) for k,v in per_f.items()} } "
              f"({ep['seconds']}s)", flush=True)

        if macro > best_macro:
            best_macro = macro
            patience = 0
            Path(a.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), a.save)
            log['best_epoch'] = epoch + 1
            log['best_macro_valid_ndcg20'] = macro
            logging.info(f'Best macro {macro} saved to {a.save}')
        else:
            patience += 1
            logging.info(f'No improvement. Early stop counter: {patience}')
            if patience >= a.early_stop:
                logging.info('Early stopping triggered.')
                break
        (Path(a.run_dir) / 'train_log.json').write_text(json.dumps(log, indent=2))
        log['elapsed_s'] = round(time.time() - t0, 1)

    log['elapsed_s'] = round(time.time() - t0, 1)
    log['best_macro_valid_ndcg20'] = best_macro
    (Path(a.run_dir) / 'train_log.json').write_text(json.dumps(log, indent=2))
    print(f'[{ds}] done best_macro={best_macro:.6f} elapsed={log["elapsed_s"]}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
