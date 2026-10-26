#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress monitor for the shared-model training and the STEP 2 finisher.

Logs a status line every `--interval` seconds (default 600 = 10 minutes) with
per-dataset epochs, best macro validation NDCG@20, recent learning rate of the
curve, throughput, and a projected completion window. Runs until the finisher has
written its status file (or both the training and finisher processes are gone).

Writes step2_shared_model/_monitor.log and prints each line to stdout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, STEP2  # noqa: E402

LOG = STEP2 / '_monitor.log'
RUNS = STEP2 / 'runs'
STATUS = STEP2 / '_FINISH_STATUS.json'

# canonical spectral-K3 item-level validation NDCG@20 reference curves:
# (epochs to best, best value) read from the audited training logs. The shared
# model sees 2x samples per epoch, so an optimistic projection halves these and a
# pessimistic one does not.
CANON_BEST_EPOCH = {'Beauty': 115, 'Cell_Phones_and_Accessories': 83,
                    'Health_and_Personal_Care': 75, 'Toys_and_Games': 102}
PATIENCE = 10
SIZE = {'Beauty': 131410, 'Cell_Phones_and_Accessories': 138681,
        'Health_and_Personal_Care': 269137, 'Toys_and_Games': 109352}
# measured seconds per epoch for the running pair, used to scale the rest
EPOCH_S = {}


def log(line):
    ts = time.strftime('%H:%M:%S')
    out = f'{ts} {line}'
    print(out, flush=True)
    with LOG.open('a') as f:
        f.write(out + '\n')


def alive(pattern):
    r = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
    return bool(r.stdout.strip())


def snapshot():
    rows = []
    for f in sorted(RUNS.glob('*/train_log.json')):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        e = d.get('epochs', [])
        if not e:
            continue
        mac = [x['macro_valid_ndcg20'] for x in e]
        sec = [x['seconds'] for x in e]
        k = min(8, len(mac))
        slope = (mac[-1] - mac[-k]) / k
        be = max(range(len(mac)), key=lambda i: mac[i]) + 1
        rows.append({'dataset': d['dataset'], 'n': len(mac), 'cur': mac[-1],
                     'best': max(mac), 'slope': slope,
                     'sec': sum(sec) / len(sec),
                     'patience': len(mac) - be, 'done': (len(mac) - be) >= PATIENCE})
    return rows


def project(rows):
    """Remaining-time estimate driven by the OBSERVED overrun factor.

    The completed runs show how far past their canonical plateau epoch the shared
    models actually went (Beauty 163/125 = 1.30, Toys 188/112 = 1.68), so the
    overrun factor is estimated from them and applied to the runs still going,
    rather than assuming a fixed number of extra epochs. Reported three ways:

      flat   each running model stops after PATIENCE more epochs
      likely canonical_best_epoch * observed_overrun + PATIENCE
      cap    runs to the hard num_epochs limit
    """
    MAX_EPOCHS = 200
    over = [r['n'] / CANON_BEST_EPOCH[r['dataset']]
            for r in rows
            if r.get('done') and CANON_BEST_EPOCH.get(r['dataset'])]
    overrun = sum(over) / len(over) if over else 1.30
    per_dataset, lo, mid, hi = {}, 0.0, 0.0, 0.0
    for r in rows:
        ds, n, sec = r['dataset'], r['n'], r['sec']
        if r.get('done'):
            lo_ds = mid_ds = hi_ds = 0.0
        else:
            nb = CANON_BEST_EPOCH.get(ds, 100)
            likely_stop = max(nb * overrun + PATIENCE, n + PATIENCE)
            lo_ds = max(n + PATIENCE - n, 0) * sec / 3600
            mid_ds = max(likely_stop - n, 0) * sec / 3600
            hi_ds = max(MAX_EPOCHS - n, 0) * sec / 3600
        per_dataset[ds] = (round(lo_ds, 2), round(mid_ds, 2), round(hi_ds, 2))
        lo, mid, hi = max(lo, lo_ds), max(mid, mid_ds), max(hi, hi_ds)
    return per_dataset, lo, mid, hi, overrun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=600)
    a = ap.parse_args()
    log('monitor started (interval %ds)' % a.interval)
    while True:
        rows = snapshot()
        parts = [f"{r['dataset'][:12]} ep{r['n']} best={r['best']:.5f} "
                 f"slope={r['slope']:+.6f} {r['sec']:.0f}s/ep" for r in rows]
        log(' | '.join(parts) if parts else 'no training logs yet')
        per, lo, mid, hi, overrun = project(rows)
        if rows:
            log(f'  projected remaining: {lo:.1f} h (flat) / {mid:.1f} h (likely) / '
                f'{hi:.1f} h ({200}-epoch cap); observed overrun factor '
                f'{overrun:.2f}x; per-dataset (flat/likely/cap): {per}')
        if STATUS.exists():
            log('finisher status file present -> monitor exiting')
            try:
                log('  ' + json.dumps(json.loads(STATUS.read_text()))[:1200])
            except Exception:
                pass
            return 0
        if not alive(r'^python3 .*xt_run_shared\.py') and \
           not alive(r'^python3 .*xt_train_shared\.py') and \
           not alive(r'[x]t_finish_step2'):
            log('training and finisher both gone without a status file -> exiting')
            return 1
        time.sleep(a.interval)


if __name__ == '__main__':
    sys.exit(main())
