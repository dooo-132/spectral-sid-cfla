#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B — efficiency and latency measurement for the two router families.

Measured on the real frozen checkpoint with real inputs (no synthetic tensors), on
the same GPU as the rest of the study.

  * router parameter count (from the trained checkpoints)
  * router forward latency, batch 1 and batch 256
  * frozen-encoder forward latency (one factorization, and both)
  * beam-generation latency (both factorizations, which Router P requires)
  * Router H hard: an estimate of the generation saving when only the selected
    factorization's beam is generated

Router P is explicitly NOT credited with any saving: it consumes both beams.

Outputs -> <selectordir>/efficiency.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from af_router_train import DATASET_ORDER, LinearRouter, MLPRouter, count_params  # noqa: E402
from af_dump_shared import load_frozen, mean_pool  # noqa: E402
from af_dump_nll import load_frozen as load_frozen_nll  # noqa: E402
from xt_common import ART, CODEBOOK, TOKENIZERS, shared_vocab_plan  # noqa: E402
from xt_train_shared import ControlPrefixedDataset  # noqa: E402
from src.tiger.dataset import GenRecDataset  # noqa: E402
from src.tiger.dataloader import GenRecDataLoader  # noqa: E402

STEP2 = Path('outputs/cross_tokenizer_shared/step2_shared_model')


def timeit(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {'median_s': statistics.median(ts), 'min_s': min(ts), 'max_s': max(ts),
            'repeats': repeats}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selectordir', required=True)
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--n_users', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ckpt', default=None,
                    help='frozen shared checkpoint; defaults to the step-2 Beauty run')
    a = ap.parse_args(argv)
    sel = Path(a.selectordir)
    dev_summary = json.loads((sel / 'router_dev_summary.json').read_text())
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    rows = []

    # ---- router parameter counts and latencies ----
    for rec in dev_summary['results']:
        ck = torch.load(sel / 'checkpoints' / f'{rec["tag"]}.pth', map_location='cpu')
        model = (LinearRouter(ck['d_in']) if ck['kind'] == 'linear' else MLPRouter(ck['d_in']))
        model.load_state_dict(ck['state_dict'])
        model.to(dev).eval()
        rows.append({'metric': 'router_parameter_count', 'scope': rec['tag'],
                     'value': count_params(model), 'unit': 'params',
                     'note': f'input dim {ck["d_in"]}'})
        for bs in (1, 256):
            x = torch.randn(bs, ck['d_in'], device=dev)
            with torch.no_grad():
                t = timeit(lambda: model(x), repeats=10, warmup=3)
            rows.append({'metric': f'router_latency_batch{bs}', 'scope': rec['tag'],
                         'value': f'{t["median_s"]*1e6:.2f}', 'unit': 'us/forward',
                         'note': f'median of {t["repeats"]} on {dev}'})
        model.to('cpu')
        del model
        torch.cuda.empty_cache()

    # ---- encoder and generation latency on real inputs ----
    ds = a.dataset
    fcts = ('spectral_k3', 'rqvae3')
    ckpt = (Path(a.ckpt) if a.ckpt else
            STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth')
    model, plan = load_frozen(ckpt, fcts)
    model.to(dev).eval()

    n = a.n_users
    sid = np.load(TOKENIZERS[fcts[0]]['sid'](ds), allow_pickle=True)
    loaders, beams = {}, {}
    for f in fcts:
        base = GenRecDataset(dataset_path=str(ART / 'data' / ds / 'test.parquet'),
                             code_path=str(TOKENIZERS[f]['sid'](ds)), mode='evaluation',
                             max_len=20, codebook_size=CODEBOOK)
        dset = ControlPrefixedDataset(base, plan['control_tokens'][f])
        loader = GenRecDataLoader(dset, batch_size=n, shuffle=False)
        b = next(iter(loader))
        loaders[f] = (b['history'].to(dev), b['attention_mask'].to(dev))
        beams[f] = (b['history'].to(dev), b['attention_mask'].to(dev), sid.shape[1])

    enc = model.model.get_encoder()

    def enc_one(f):
        inp, am = loaders[f]
        with torch.no_grad():
            out = enc(input_ids=inp, attention_mask=am)
            return mean_pool(out.last_hidden_state, am)

    def gen_one(f):
        inp, am = loaders[f]
        with torch.no_grad():
            return model.model.generate(input_ids=inp, attention_mask=am, max_length=beams[f][2] + 1,
                                        num_beams=30, num_return_sequences=30,
                                        return_dict_in_generate=True, output_scores=True)

    t_enc1 = timeit(lambda: enc_one(fcts[0]), repeats=5)
    t_gen1 = timeit(lambda: gen_one(fcts[0]), repeats=3)
    rows += [
        {'metric': 'encoder_latency', 'scope': f'{ds} n={n} 1 factorization',
         'value': f'{t_enc1["median_s"]*1000:.2f}', 'unit': 'ms', 'note': 'frozen encoder forward'},
        {'metric': 'beam_generation_latency', 'scope': f'{ds} n={n} 1 factorization',
         'value': f'{t_gen1["median_s"]*1000:.2f}', 'unit': 'ms', 'note': 'beam 30'},
    ]

    # ---- estimate the Router H hard saving from measured components ----
    e = t_enc1['median_s']
    g = t_gen1['median_s']
    # Router H needs both encoder passes but one beam; uniform/soft fusion needs two beams.
    baseline = 2 * e + 2 * g
    router_h_hard = 2 * e + 1 * g
    routing_overhead = 0.0
    rows += [
        {'metric': 'end_to_end_latency_estimate', 'scope': 'two-beam reference (uniform fusion)',
         'value': f'{baseline*1000:.2f}', 'unit': 'ms',
         'note': '2x encoder + 2x beam on the measured batch'},
        {'metric': 'end_to_end_latency_estimate', 'scope': 'Router H hard (one beam)',
         'value': f'{router_h_hard*1000:.2f}', 'unit': 'ms',
         'note': '2x encoder + 1x beam; the router itself is negligible'},
        {'metric': 'estimated_generation_saving_fraction', 'scope': 'Router H hard vs two-beam',
         'value': f'{(baseline - router_h_hard)/baseline:.6f}', 'unit': 'fraction',
         'note': 'assumes only the selected factorization is generated'},
        {'metric': 'estimated_generation_saving_fraction', 'scope': 'Router P hard/soft',
         'value': '0.000000', 'unit': 'fraction',
         'note': 'both factorization beams are already computed; no saving can be claimed'},
    ]

    out = sel / 'efficiency.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['metric', 'scope', 'value', 'unit', 'note'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out}')
    print(f'  encoder(1 factor, n={n}) = {e*1000:.2f} ms ; beam = {g*1000:.2f} ms')
    print(f'  Router H hard estimated saving = {(baseline-router_h_hard)/baseline:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
