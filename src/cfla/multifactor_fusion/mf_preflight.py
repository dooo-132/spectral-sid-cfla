#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE D preflight — OOM preflight BEFORE any 5-factor NLL scoring (section 18).

Small-scale rehearsal on Beauty with <= 64 users, measuring the actual candidate
union size, the number of candidate sequences, peak allocated/reserved VRAM and
free/used VRAM before and after.  Also probes the largest sweep of factorizations
(5) so the measurement bounds the real run.

Hard rule (sections 18, 56): if peak VRAM exceeds 90% of total VRAM, scoring does
NOT start, the run returns STOPPED_DUE_TO_MEMORY_RISK, and no batch size, beam or
candidate count is adjusted to get under the line.

Writes `OOM_PREFLIGHT.md` and appends to `gpu_memory_log.csv`.

Usage
  python3 mf_preflight.py --users 64
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import DATASETS, FACTORIZATIONS, OUT, S4, M5, load_sid, \
    sha256_file, sid_path, sid_tokens_for_rows, split_user_ids, write_json, \
    load_beam_cell, build_union
from mf_models import (build_history_tokens, candidate_total_nll,
                       encoder_input_for, load_model_for)
from mf_candidate_nll import BATCH_USERS, CAND_CHUNK, CONFIG_REVISION, union_dir

GPU_LOG = OUT / 'gpu_memory_log.csv'
GPU_COLS = ['timestamp', 'stage', 'dataset', 'split', 'gpu_name', 'mem_total_mib',
            'mem_used_mib', 'mem_free_mib', 'util_pct', 'external_procs_mib']


def nvidia_smi():
    try:
        q = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
        line = q.stdout.strip().splitlines()[0]
        name, tot, used, free, util = [x.strip() for x in line.split(',')]
        return {'gpu_name': name, 'mem_total_mib': float(tot), 'mem_used_mib': float(used),
                'mem_free_mib': float(free), 'util_pct': float(util)}
    except Exception as e:                                   # pragma: no cover
        return {'gpu_name': f'unknown ({e})', 'mem_total_mib': float('nan'),
                'mem_used_mib': float('nan'), 'mem_free_mib': float('nan'),
                'util_pct': float('nan')}


def log_gpu(stage, dataset='', split=''):
    d = nvidia_smi()
    row = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'stage': stage,
           'dataset': dataset, 'split': split, **d,
           'external_procs_mib': d['mem_used_mib']}
    GPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not GPU_LOG.exists()
    with GPU_LOG.open('a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=GPU_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--split', default='valid')
    ap.add_argument('--users', type=int, default=64)
    ap.add_argument('--pool', default='multi5', choices=['spectral4', 'multi5'])
    ap.add_argument('--chunk', type=int, default=CAND_CHUNK)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)

    import torch
    before = log_gpu('preflight_before', a.dataset, a.split)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- union side ---------------------------------------------------------
    cells = {f: load_beam_cell(f, a.dataset, a.split) for f in FACTORIZATIONS}
    union = build_union(cells, user_ids=split_user_ids(a.dataset, a.split))
    f_list = M5 if a.pool == 'multi5' else S4
    fi = [union['f_order'].index(f) for f in f_list]
    mask = union['items'] >= 0
    sizes = mask.sum(axis=1)
    n_users = min(a.users, union['n_users'])
    sel = np.arange(n_users)
    pairs = int(mask[sel].sum())
    seqs = pairs * len(f_list)

    # --- scoring side -------------------------------------------------------
    cache = {}
    per_f = {}
    t0 = time.time()
    for f in f_list:
        sid = load_sid(f, a.dataset)
        bundle = load_model_for(f, a.dataset, device=a.device, cache=cache)
        h, m = build_history_tokens(a.dataset, a.split, sid_path(f, a.dataset))
        h, m = encoder_input_for(bundle, f, h, m)
        # rows of this user block, flattened in the union's own order
        cand_rows = union['items'][sel].reshape(-1)
        keep = cand_rows >= 0
        owner = np.repeat(np.arange(n_users), union['items'].shape[1])[keep]
        rows = cand_rows[keep] - 1                       # canonical -> SID row
        toks = sid_tokens_for_rows(sid, rows)
        torch.cuda.reset_peak_memory_stats()
        nll = candidate_total_nll(bundle['model'], h[:n_users], m[:n_users], toks,
                                  owner, device=a.device, batch_size=a.chunk)
        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
        peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
        per_f[f] = {'sequences': int(len(toks)), 'peak_allocated_mib': peak_alloc,
                    'peak_reserved_mib': peak_res,
                    'finite_fraction': float(np.isfinite(nll).mean()),
                    'mean_nll': float(np.nanmean(nll)),
                    'checkpoint': bundle['checkpoint'],
                    'checkpoint_sha256': sha256_file(bundle['checkpoint'])}
        del toks, nll
        torch.cuda.empty_cache()
        print(f'  {f:12s} seqs={len(rows):6d} peak_alloc={peak_alloc:8.1f} MiB '
              f'peak_res={peak_res:8.1f} MiB', flush=True)

    after = log_gpu('preflight_after', a.dataset, a.split)
    peak_alloc = max(v['peak_allocated_mib'] for v in per_f.values())
    peak_res = max(v['peak_reserved_mib'] for v in per_f.values())
    total = before['mem_total_mib']
    used_after = after['mem_used_mib']
    frac_peak = peak_res / total if total else float('nan')
    frac_used = used_after / total if total else float('nan')

    worst_f = max(per_f, key=lambda f: per_f[f]['peak_reserved_mib'])
    risk = bool((not np.isfinite(frac_peak)) or frac_peak > 0.90
                or frac_used > 0.90)
    verdict = 'STOPPED_DUE_TO_MEMORY_RISK' if risk else 'PROCEED'

    report = {
        'dataset': a.dataset, 'split': a.split, 'pool': a.pool, 'users': n_users,
        'candidate_union_size_mean': float(sizes[sel].mean()),
        'candidate_union_size_max': int(sizes[sel].max()),
        'candidate_factorisation_pairs': pairs,
        'candidate_sequences': seqs,
        'cand_chunk_predeclared': a.chunk,
        'batch_users_predeclared': BATCH_USERS,
        'config_revision': CONFIG_REVISION,
        'factorizations': list(f_list),
        'per_factorization': per_f,
        'peak_allocated_mib': peak_alloc, 'peak_reserved_mib': peak_res,
        'worst_factorization': worst_f,
        'gpu_total_mib': total,
        'gpu_used_before_mib': before['mem_used_mib'],
        'gpu_used_after_mib': used_after,
        'gpu_free_after_mib': after['mem_free_mib'],
        'vram_peak_fraction_of_total': frac_peak,
        'vram_used_fraction_after': frac_used,
        'threshold_fraction': 0.90,
        'memory_risk': risk,
        'verdict': verdict,
        'elapsed_s': round(time.time() - t0, 1),
        'external_gpu_processes': subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
            capture_output=True, text=True).stdout.strip(),
        'note': ('The 90% test is applied to BOTH the torch peak-reserved fraction and '
                 'the device used fraction after the rehearsal; either one trips the stop.'),
    }
    write_json(OUT / 'OOM_PREFLIGHT.json', report)

    md = [f'# OOM_PREFLIGHT — {a.dataset} {a.split} ({a.pool})', '',
          f'Rehearsal of the full 5-factorization candidate-NLL forward pass on '
          f'**{n_users} users** of `{a.dataset}`/{a.split}, using the same predeclared '
          f'chunk size ({a.chunk} candidate sequences per forward pass) as the real run.',
          '', '## Candidate union', '',
          f'| quantity | value |', '|---|---|',
          f'| candidate union size, mean | {report["candidate_union_size_mean"]:.2f} |',
          f'| candidate union size, max | {report["candidate_union_size_max"]} |',
          f'| candidate × factorization pairs | {pairs} |',
          f'| teacher-forced sequences scored | {seqs} |',
          f'| factorizations in this rehearsal | {len(f_list)} ({", ".join(f_list)}) |',
          '', '## Per-factorization VRAM', '',
          '| factorization | sequences | peak allocated (MiB) | peak reserved (MiB) | finite |',
          '|---|---|---|---|---|']
    for f, v in per_f.items():
        md.append(f'| {f} | {v["sequences"]} | {v["peak_allocated_mib"]:.1f} | '
                  f'{v["peak_reserved_mib"]:.1f} | {v["finite_fraction"]:.6f} |')
    md += ['', '## Device state', '',
           f'| quantity | value |', '|---|---|',
           f'| GPU | {before["gpu_name"]} |',
           f'| total VRAM (MiB) | {total:.0f} |',
           f'| used before (MiB) | {before["mem_used_mib"]:.0f} |',
           f'| used after (MiB) | {used_after:.0f} |',
           f'| free after (MiB) | {after["mem_free_mib"]:.0f} |',
           f'| torch peak reserved (MiB) | {peak_res:.1f} |',
           f'| peak reserved / total | **{frac_peak:.4f}** |',
           f'| used / total after | **{frac_used:.4f}** |',
           f'| threshold | 0.90 |',
           '', f'External GPU processes at preflight time:',
           f'```\n{report["external_gpu_processes"]}\n```',
           '', '## Verdict', '', f'```\n{verdict}\n```', '',
           ('Peak VRAM stays below 90% of total VRAM on both measures, so the full '
            'validation scoring may proceed at the predeclared chunk size.'
            if not risk else
            'Peak VRAM exceeds 90% of total VRAM. Per sections 18 and 56 the run stops '
            'here: no batch size, beam size, candidate count, precision or architecture '
            'change is made, and no full scoring is started.')]
    (OUT / 'OOM_PREFLIGHT.md').write_text('\n'.join(md) + '\n')
    print(f'\npeak reserved {peak_res:.1f} MiB / {total:.0f} MiB = {frac_peak:.4f} '
          f'(threshold 0.90)  ->  {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
