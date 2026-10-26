#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_m5_cfla.py -- turn the M5 CFLA rows of `m5_rq5/{VALID,TEST,BOOTSTRAP}.csv`
from `BLOCKED_NO_CANDIDATE_NLL` into measured numbers, WITHOUT modifying the
frozen driver.

WHY A WRAPPER AND NOT AN EDIT
-----------------------------
`build_m5_rq5.py` emits its CFLA rows through a hard-wired code path:

    evaluate_strategies(...)  ->  ...  blocked('cfla')        # line ~954
    run(...)                  ->  cfla_rows_blocked(...)      # line ~1756
    skipped_entries(...)      ->  {'what': 'cfla on the m5 scope',
                                   'status': ST_BLOCKED_NO_NLL, ...}

It never opens a candidate-NLL parquet.  Re-running it therefore cannot turn
those rows into measurements, no matter what exists on disk.  The driver is a
published artifact whose sha256 is recorded in the run metadata it produced, so
it is NOT edited here.  Instead this wrapper

  * imports it as a module (`sys.dont_write_bytecode = True` first, so nothing is
    written into any read-only root),
  * replaces exactly three things in memory --
        `evaluate_strategies`  (inject the measured CFLA strategy),
        `bootstrap_pairs`      (append the `cfla_vs_uniform` bootstrap block),
        `cfla_rows_blocked`    (emit nothing once CFLA is measured) and
        `skipped_entries`      (drop the now-false "cfla is blocked" claim),
  * and otherwise runs the frozen pipeline unchanged: same union construction,
    same VALID-only selection, same metrics, same merge guard, same CSV contract.

WHAT THE MEASURED CFLA IS
-------------------------
    zNLL_F(u,i) = (NLL_F(u,i) - mu_F) / sigma_F          (mu, sigma fitted on VALID)
    score(u,i)  = logsumexp_F( -zNLL_F(u,i) )

`NLL_F` and the scalers come from `m5_rq5/candidate_nll/` (produced by
`score_m5_candidate_nll.py`), the aggregation is `build_m5_rq5.cfla_scores_from_nll`
(the frozen driver's own function), and the ranking is
`fusion_audit.per_user_ndcg` applied to the CFLA score as a single-arm tensor --
so no metric code is re-implemented.

ALIGNMENT IS PROVED, NOT ASSUMED
--------------------------------
The parquet's rows must equal the union's own rows, in order, on `(user_row, pos,
candidate_item)`.  The wrapper identifies the dataset by that exact match (with the
driver's own call order as a hint, never as the only evidence) and hard-fails on a
mismatch, a duplicate key, a missing arm column, a non-finite NLL or a degenerate
scaler.

REVISION DISCIPLINE
-------------------
The M5 aggregate refuses two DIFFERENT measurements of one cell, which is exactly
what a status revision produces.  This wrapper therefore uses the driver's own
documented mechanism: every earlier run whose m5-scope `cfla` row carries a
different status is declared SUPERSEDED with a written reason, in
`m5_rq5/SUPERSEDED.json`.  The run directories stay on disk, the merge guard is
NOT weakened, and the guard still refuses a merge that would drop a row.

Usage
    python3 build_m5_cfla.py --check          # inputs + alignment, writes nothing
    python3 build_m5_cfla.py --dry-run        # full CPU pipeline, writes nothing
    python3 build_m5_cfla.py --datasets <4> --tag m5rq5cfla
"""
from __future__ import annotations

import os as _os
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[3])
_ART_ROOT = _os.environ.get('CFLA_ARTIFACT_ROOT') or _os.path.join(_REPO_ROOT, 'artifacts')
_UP_ROOT = _os.environ.get('CFLA_UPSTREAM_ROOT') or _ART_ROOT
import sys as _sys
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from src.utils import repo_paths as rp  # noqa: E402
import sys as _sys

_sys.dont_write_bytecode = True

import argparse                                                        # noqa: E402
import csv                                                             # noqa: E402
import json                                                            # noqa: E402
import os                                                              # noqa: E402
import time                                                            # noqa: E402
from datetime import datetime, timezone                                # noqa: E402
from pathlib import Path                                               # noqa: E402

import numpy as np                                                     # noqa: E402

REPO = Path(_REPO_ROOT + '')
OUT = rp.OUTPUT_ROOT / 'final_hyperparam_efficiency'
DRIVERS = Path(__file__).resolve().parent
M5DIR = OUT / 'm5_rq5'
CAND = M5DIR / 'candidate_nll'
SCALERS_PATH = CAND / 'validation_nll_scalers.json'
DATASETS = ['Beauty', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games']
FACTORIZATIONS = ['spectral_k2', 'spectral_k3', 'spectral_k4', 'spectral_k5',
                  'rqvae5']
LABEL = {'spectral_k2': 'K2', 'spectral_k3': 'K3', 'spectral_k4': 'K4',
         'spectral_k5': 'K5', 'rqvae5': 'RQ5'}
CFLA_STATUS = 'CFLA_MEASURED_FROM_M5_CANDIDATE_NLL'

if str(DRIVERS) not in _sys.path:
    _sys.path.insert(0, str(DRIVERS))
import build_m5_rq5 as bm                                             # noqa: E402


# ===========================================================================
# inputs
# ===========================================================================
def parquet_path(ds, split) -> Path:
    return CAND / ds / f'{split}_m5_candidate_nll.parquet'


def load_inputs(ds, split, arms, cache):
    """Read the candidate-NLL parquet + the VALID-fitted scalers for one cell."""
    import pandas as pd
    key = (ds, split)
    if key in cache:
        return cache[key]
    pq = parquet_path(ds, split)
    if not pq.exists():
        return None
    cols = ['dataset', 'split', 'user_row', 'pos', 'candidate_item'] + \
           [f'NLL_{LABEL[f]}' for f in arms]
    df = pd.read_parquet(pq, columns=cols)
    if SCALERS_PATH.exists():
        sc = json.loads(SCALERS_PATH.read_text())
    else:
        return {'error': f'missing {SCALERS_PATH}', 'parquet': str(pq)}
    blk = (sc.get('pools') or {}).get('m5', {}).get(ds)
    if not blk:
        return {'error': f'{SCALERS_PATH} has no pools.m5.{ds} block',
                'parquet': str(pq)}
    scalers = {}
    for f in arms:
        L = LABEL[f]
        if L not in blk:
            return {'error': f'scalers for {ds}/{f} ({L}) missing', 'parquet': str(pq)}
        if f'NLL_{L}' not in df.columns:
            return {'error': f'column NLL_{L} missing from {pq}', 'parquet': str(pq)}
        scalers[f] = (float(blk[L]['mean']), float(blk[L]['std']))
    out = {'df': df, 'scalers': scalers, 'parquet': str(pq), 'arms': list(arms),
           'scalers_path': str(SCALERS_PATH)}
    cache[key] = out
    return out


def cfla_rows_for_union(inputs, U, arms, ds, split):
    """Per-union-row CFLA score, PROVED to belong to the union's own rows."""
    import pandas as pd
    df = inputs['df']
    items_pad = np.asarray(U[0])
    rows_i, rows_p = np.where(items_pad >= 0)
    cands = np.asarray(items_pad[rows_i, rows_p], dtype=np.int64)
    n = len(cands)
    if len(df) != n:
        raise AssertionError(
            f'{ds}/{split}: the candidate-NLL parquet has {len(df)} rows but the '
            f'union has {n} -- refusing to align by position')
    got_u = df['user_row'].to_numpy(dtype=np.int64)
    got_p = df['pos'].to_numpy(dtype=np.int64)
    got_c = df['candidate_item'].to_numpy(dtype=np.int64)
    if not (np.array_equal(got_u, rows_i) and np.array_equal(got_p, rows_p)
            and np.array_equal(got_c, cands)):
        bad = int((got_c != cands).sum())
        raise AssertionError(
            f'{ds}/{split}: the parquet row order does not equal the union row order '
            f'({bad} candidate mismatches); the CFLA score cannot be scattered onto '
            f'the union safely')
    got_ds = set(df['dataset'].unique().tolist())
    if got_ds != {ds}:
        raise AssertionError(f'{ds}/{split}: parquet records dataset(s) {got_ds}')
    nll = {f: df[f'NLL_{LABEL[f]}'].to_numpy(dtype=np.float64) for f in arms}
    for f, x in nll.items():
        if not np.isfinite(x).all():
            raise AssertionError(f'{ds}/{split}/{f}: non-finite NLL in the parquet')
    cf = bm.cfla_scores_from_nll(nll, inputs['scalers'], list(arms))
    cf = np.asarray(cf, dtype=np.float64)
    if cf.shape != (n,) or not np.isfinite(cf).all():
        raise AssertionError(f'{ds}/{split}: CFLA vector {cf.shape} is not finite')
    mat = np.zeros(items_pad.shape, dtype=np.float64) - 1e30
    mat[rows_i, rows_p] = cf
    return cf, mat


def per_user_from_score(fa, U, mat):
    """`fusion_audit.per_user_ndcg` on the CFLA score as a single-arm tensor."""
    items_pad, _score_tensor, _target, target_pos, valid_mask = U
    n = items_pad.shape[0]
    w = np.ones((n, 1), dtype=np.float64)
    return fa.per_user_ndcg(items_pad, mat[:, :, None], target_pos, valid_mask, w)


# ===========================================================================
# the in-memory patch surface
# ===========================================================================
STATE = {'datasets': list(DATASETS), 'arms': list(FACTORIZATIONS), 'cache': {},
         'dataset_hint': None, 'log': [], 'alignment': {}, 'available': {},
         'st_ok': bm.ST_OK, 'st_blocked': bm.ST_BLOCKED_NO_NLL}


def probe_availability():
    """Per dataset: can a measured CFLA be produced?  Recorded, never guessed."""
    avail = {}
    for ds in STATE['datasets']:
        why = []
        for split in ('valid', 'test'):
            p = parquet_path(ds, split)
            if not p.exists():
                why.append(f'missing {p}')
        if not SCALERS_PATH.exists():
            why.append(f'missing {SCALERS_PATH}')
        avail[ds] = {'available': not why, 'reasons': why,
                     'parquet_valid': str(parquet_path(ds, 'valid')),
                     'parquet_test': str(parquet_path(ds, 'test')),
                     'scalers': str(SCALERS_PATH)}
    STATE['available'] = avail
    return avail


def patched_load_scope_inputs(fa, dataset, affinity, split, scope):
    """Record which dataset the driver is working on (a hint, cross-checked)."""
    if scope == 'm5':
        STATE['dataset_hint'] = dataset
    return _ORIG_LOAD_SCOPE_INPUTS(fa, dataset, affinity, split, scope)


def identify_dataset(U, split):
    """Whose parquet is this union?  Decided by an EXACT row match, not by order."""
    items_pad = np.asarray(U[0])
    rows_i, rows_p = np.where(items_pad >= 0)
    cands = np.asarray(items_pad[rows_i, rows_p], dtype=np.int64)
    order = [STATE['dataset_hint']] + [d for d in STATE['datasets']
                                       if d != STATE['dataset_hint']]
    for ds in order:
        if ds is None:
            continue
        inputs = load_inputs(ds, split, STATE['arms'], STATE['cache'])
        if not inputs or 'error' in inputs:
            continue
        df = inputs['df']
        if len(df) != len(cands):
            continue
        if (np.array_equal(df['user_row'].to_numpy(dtype=np.int64), rows_i)
                and np.array_equal(df['candidate_item'].to_numpy(dtype=np.int64),
                                   cands)):
            return ds, inputs
    return None, None


def measured_cfla(fa, unions, arms):
    """Build the CFLA strategy entry for BOTH splits, or raise."""
    per_user, detail, algn = {}, {}, {}
    for split in ('valid', 'test'):
        U = unions[split]
        ds, inputs = identify_dataset(U, split)
        if ds is None:
            raise AssertionError(
                f'{split}: no candidate-NLL parquet matches the union rows of this '
                f'cell (hint={STATE["dataset_hint"]!r}); refusing to score CFLA')
        cf, mat = cfla_rows_for_union(inputs, U, arms, ds, split)
        per_user[split] = per_user_from_score(fa, U, mat)
        algn[split] = {'dataset': ds, 'n_rows': int(cf.shape[0]),
                       'parquet': inputs['parquet'],
                       'cf_mean': float(cf.mean()), 'cf_std': float(cf.std(ddof=0)),
                       'cf_min': float(cf.min()), 'cf_max': float(cf.max())}
        detail[f'nll_parquet_{split}'] = inputs['parquet']
    detail.update({
        'aggregation': 'score(u,i) = logsumexp_F(-zNLL_F(u,i)); zNLL_F = '
                       '(NLL_F - mu_F)/sigma_F',
        'aggregation_function': 'build_m5_rq5.cfla_scores_from_nll (frozen driver)',
        'ranking_function': 'fusion_audit.per_user_ndcg, single-arm tensor',
        'scalers': str(SCALERS_PATH), 'scalers_fit_split': 'valid',
        'test_used_for_fitting': False, 'arms': list(arms),
        'alignment': algn})
    return {'status': bm.ST_OK, 'missing_input': '', 'producing_command': '',
            'detail': detail, 'per_user': per_user}


def patched_evaluate_strategies(fa, p5, unions, confs, q_norm, scope, strategy_status):
    out = _ORIG_EVALUATE(fa, p5, unions, confs, q_norm, scope, strategy_status)
    if scope != 'm5':
        return out
    ds = STATE['dataset_hint']
    avail = STATE['available'].get(ds, {'available': False, 'reasons': ['unknown']})
    if not avail['available']:
        out['cfla'] = {'status': bm.ST_BLOCKED_NO_NLL,
                       'missing_input': ' ; '.join(avail['reasons']),
                       'producing_command': bm.cfla_producing_command(STATE['datasets']),
                       'detail': {}, 'per_user': {'valid': None, 'test': None}}
        STATE['log'].append({'event': 'cfla_blocked', 'dataset': ds,
                             'reasons': avail['reasons']})
        return out
    entry = measured_cfla(fa, unions, STATE['arms'])
    entry['detail']['dataset'] = ds
    entry['detail']['measured_by'] = 'build_m5_cfla.py (wrapper; frozen driver ' \
                                     'unmodified)'
    out['cfla'] = entry
    STATE['alignment'][ds] = entry['detail']['alignment']
    STATE['log'].append({'event': 'cfla_measured', 'dataset': ds,
                         'alignment': entry['detail']['alignment']})
    print(f'    cfla MEASURED ({ds}): '
          + ', '.join(f"{s}: mean={entry['per_user'][s]['NDCG@20'].mean():.6f} "
                      f"n={len(entry['per_user'][s]['NDCG@20'])}"
                      for s in ('valid', 'test')), flush=True)
    return out


def patched_bootstrap_pairs(p5, run_id, dataset, affinity, scope, res):
    rows = _ORIG_BOOTSTRAP(p5, run_id, dataset, affinity, scope, res)
    if scope != 'm5':
        return rows
    cfla = res.get('cfla')
    if not cfla or cfla['per_user']['test'] is None:
        return rows
    a = cfla['per_user']['test']
    b = res['uniform']['per_user']['test']
    for key in bm.METRIC_KEYS:
        r = p5.paired_bootstrap(a[key], b[key])
        lo, hi = float(r['ci_lo']), float(r['ci_hi'])
        pct = r.get('pct_change')
        rows.append({
            'run_id': run_id, 'dataset': dataset, 'affinity': affinity,
            'scope': scope, 'comparison': 'cfla_vs_uniform', 'strategy_a': 'cfla',
            'strategy_b': 'uniform', 'metric': key,
            'a_value': repr(float(a[key].mean())), 'b_value': repr(float(b[key].mean())),
            'delta': repr(float(r['delta'])),
            'pct_change': '' if pct is None else repr(float(pct)),
            'ci_lo': repr(lo), 'ci_hi': repr(hi), 'p_value': repr(float(r['p_value'])),
            'n_boot': p5.N_BOOT, 'seed': p5.RNG_SEED, 'n_users': int(r['n_users']),
            'ci95_excludes_zero': bool(lo > 0.0 or hi < 0.0),
            'status': bm.ST_OK, 'missing_input': '',
            'notes': 'measured by build_m5_cfla.py; ' + str(r.get('note', '') or '')})
    print(f'    bootstrap cfla_vs_uniform added: {len(bm.METRIC_KEYS)} metric row(s) '
          f'({dataset}); n_users={int(rows[-1]["n_users"])}', flush=True)
    return rows


def patched_cfla_rows_blocked(run_id, dataset, affinity, scope, status, missing,
                              command):
    avail = STATE['available'].get(dataset)
    if avail and avail['available']:
        # CFLA is measured for this dataset in this run: a blocked row would be a
        # false statement about a computed number.
        return []
    return _ORIG_CFLA_ROWS_BLOCKED(run_id, dataset, affinity, scope, status, missing,
                                   command)


def patched_skipped_entries(args, per_dataset):
    out = _ORIG_SKIPPED(args, per_dataset)
    kept, replaced = [], []
    for e in out:
        if e.get('what') == 'cfla on the m5 scope':
            replaced.append(e)
            if any(not v['available'] for v in STATE['available'].values()):
                e = dict(e)
                e['reason'] = ('CFLA needs NLL_F(u,i) for all five factorizations over '
                               'the M5 union plus VALID-fitted scalers. ' + e['reason'])
                kept.append(e)
            continue
        kept.append(e)
    kept.append({
        'what': 'cfla on the m5 scope',
        'status': CFLA_STATUS,
        'reason': 'CFLA was measured by build_m5_cfla.py from '
                  f'{CAND}/<dataset>/<split>_m5_candidate_nll.parquet and the '
                  f'VALID-fitted scalers in {SCALERS_PATH}. The frozen driver '
                  'build_m5_rq5.py emits cfla through a hard-wired '
                  'BLOCKED_NO_CANDIDATE_NLL path and cannot see these inputs; it '
                  'was not modified. This run therefore replaces that status with '
                  'measured rows.',
        'replaced_entries': len(replaced),
        'alignment': {k: {'n_rows': v['valid']['n_rows'],
                          'parquet': v['valid']['parquet']}
                      for k, v in STATE['alignment'].items()}})
    return kept


def patched_merge_aggregate(dest, fields, key_fields, run_id):
    """Guard the merge: the wrapper may change ONLY the cfla rows.

    Runs BEFORE the frozen merge touches the aggregate.  Every non-cfla cell the
    new run shares with an earlier run must be bit-identical (`bm._same_measurement`
    ignores only `run_id`), otherwise the wrapper would be silently rewriting
    measured numbers it has no business touching.  The cfla cells are expected to
    differ (BLOCKED -> OK) and are reported instead.
    """
    new_dir = M5DIR / 'runs' / run_id
    if new_dir.is_dir() and dest.exists() is not None:
        new_rows = _read_csv(new_dir / dest.name)
        new_by_key = {}
        for r in new_rows:
            k = tuple(r.get(x, '') for x in key_fields)
            if k in new_by_key:
                raise AssertionError(f'{dest.name}: the new run records key {k} twice')
            new_by_key[k] = r
        sup = {}
        sp = M5DIR / 'SUPERSEDED.json'
        if sp.exists():
            sup = (json.loads(sp.read_text()) or {}).get('superseded', {}) or {}
        diffs, cfla_changed, compared = [], [], 0
        runs = M5DIR / 'runs'
        for d in sorted(p for p in runs.iterdir() if p.is_dir()):
            if d.name == run_id or d.name in sup:
                continue
            for r in _read_csv(d / dest.name):
                k = tuple(r.get(x, '') for x in key_fields)
                if k not in new_by_key:
                    continue
                compared += 1
                if bm._same_measurement(new_by_key[k], r):
                    continue
                if str(k).find('cfla') >= 0 or r.get('strategy') == 'cfla' \
                        or r.get('comparison', '').startswith('cfla'):
                    cfla_changed.append({'key': list(k), 'run': d.name,
                                         'old_status': r.get('status'),
                                         'new_status': new_by_key[k].get('status')})
                else:
                    diffs.append({'key': list(k), 'run': d.name,
                                  'old': {x: r.get(x) for x in fields
                                          if r.get(x) != new_by_key[k].get(x)},
                                  'new': {x: new_by_key[k].get(x) for x in fields
                                          if r.get(x) != new_by_key[k].get(x)}})
        STATE['premerge'] = {'dest': str(dest), 'rows_compared': compared,
                             'non_cfla_differences': diffs[:20],
                             'n_non_cfla_differences': len(diffs),
                             'cfla_status_changes': cfla_changed[:20],
                             'n_cfla_status_changes': len(cfla_changed)}
        print(f'  [wrapper guard] {dest.name}: compared {compared} shared cell(s) '
              f'with earlier runs; non-cfla differences={len(diffs)}, '
              f'cfla status changes={len(cfla_changed)}', flush=True)
        if diffs:
            raise AssertionError(
                f'{dest.name}: the CFLA-injected run changes {len(diffs)} NON-cfla '
                f'cell(s) vs an earlier run, e.g. {diffs[0]}. The wrapper must only '
                f'add the CFLA strategy; refusing to merge. The frozen guard is '
                f'untouched and the aggregate has not been modified.')
    return _ORIG_MERGE(dest, fields, key_fields, run_id)


_ORIG_LOAD_SCOPE_INPUTS = None
_ORIG_EVALUATE = None
_ORIG_BOOTSTRAP = None
_ORIG_CFLA_ROWS_BLOCKED = None
_ORIG_SKIPPED = None
_ORIG_MERGE = None


def install_patches():
    global _ORIG_LOAD_SCOPE_INPUTS, _ORIG_EVALUATE, _ORIG_BOOTSTRAP
    global _ORIG_CFLA_ROWS_BLOCKED, _ORIG_SKIPPED, _ORIG_MERGE
    _ORIG_LOAD_SCOPE_INPUTS = bm.load_scope_inputs
    _ORIG_EVALUATE = bm.evaluate_strategies
    _ORIG_BOOTSTRAP = bm.bootstrap_pairs
    _ORIG_CFLA_ROWS_BLOCKED = bm.cfla_rows_blocked
    _ORIG_SKIPPED = bm.skipped_entries
    _ORIG_MERGE = bm.merge_aggregate
    bm.load_scope_inputs = patched_load_scope_inputs
    bm.evaluate_strategies = patched_evaluate_strategies
    bm.bootstrap_pairs = patched_bootstrap_pairs
    bm.cfla_rows_blocked = patched_cfla_rows_blocked
    bm.skipped_entries = patched_skipped_entries
    bm.merge_aggregate = patched_merge_aggregate
    return ['bm.load_scope_inputs (hint only)', 'bm.evaluate_strategies',
            'bm.bootstrap_pairs', 'bm.cfla_rows_blocked', 'bm.skipped_entries',
            'bm.merge_aggregate (pre-merge non-cfla equality guard)']


# ===========================================================================
# supersede discipline
# ===========================================================================
def cfla_status_by_run():
    runs = M5DIR / 'runs'
    out = {}
    if not runs.is_dir():
        return out
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        rows = [r for r in _read_csv(d / 'VALID.csv')
                if r.get('scope') == 'm5' and r.get('strategy') == 'cfla']
        if rows:
            out[d.name] = sorted({(r.get('dataset'), r.get('status')) for r in rows})
    return out


def _read_csv(p: Path):
    if not p.exists():
        return []
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plan_supersede(a):
    """Which earlier runs recorded a DIFFERENT cfla status than this run will write?"""
    want = {ds: (bm.ST_OK if STATE['available'][ds]['available']
                 else bm.ST_BLOCKED_NO_NLL) for ds in a.datasets}
    supersede, agree = [], []
    for run, entries in cfla_status_by_run().items():
        differs = [f'{ds}={st}' for ds, st in entries
                   if ds in want and st != want[ds]]
        if differs:
            supersede.append({'run': run, 'differs': differs})
        else:
            agree.append(run)
    return supersede, agree, want


def record_supersede(supersede, want, dry_run):
    if not supersede:
        return None
    ids = [s['run'] for s in supersede]
    detail = '; '.join(f"{s['run']}: {','.join(s['differs'])}" for s in supersede)
    reason = (
        'Status revision with measured numbers (build_m5_cfla.py): the M5 CFLA rows '
        'in these runs carry BLOCKED_NO_CANDIDATE_NLL because the five-arm '
        'candidate-NLL parquets and the VALID-fitted scalers did not exist when they '
        f'ran. They now do, so this run writes status=OK for '
        f'{", ".join(sorted(want))}. The merge guard refuses two DIFFERENT '
        'measurements of one cell, so these runs are excluded from the aggregate as '
        'an intentional revision; every run directory stays on disk and the guard\'s '
        'drop-a-row check still applies. Superseded: ' + detail)
    sp = M5DIR / 'SUPERSEDED.json'
    cur = json.loads(sp.read_text()) if sp.exists() else {'superseded': {}}
    cur.setdefault('superseded', {})
    for i in ids:
        cur['superseded'][i] = {
            'reason': reason,
            'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'superseded_by': 'build_m5_cfla.py'}
    if not dry_run:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(cur, indent=2) + '\n', encoding='utf-8')
    return {'ids': ids, 'reason': reason, 'dry_run': bool(dry_run)}


# ===========================================================================
# check mode
# ===========================================================================
def check_mode(a, fa):
    print('=' * 78)
    print('build_m5_cfla.py -- CHECK (CPU only; writes nothing)')
    print('=' * 78)
    avail = probe_availability()
    ok = True
    for ds, v in avail.items():
        print(f'  {ds:32s} available={v["available"]}'
              + ('' if v['available'] else f'  reasons={v["reasons"]}'))
        ok = ok and v['available']
    print()
    print('  the check verifies row alignment by REBUILDING the M5 union and '
          'comparing it to the parquet:')
    for ds in a.datasets:
        if not avail[ds]['available']:
            continue
        for split in ('valid', 'test'):
            U = build_union_only(fa, ds, split, a.affinity)
            inputs = load_inputs(ds, split, STATE['arms'], STATE['cache'])
            cf, _mat = cfla_rows_for_union(inputs, U, STATE['arms'], ds, split)
            print(f'    {ds:32s} {split:5s} rows={cf.shape[0]:9d} '
                  f'cf_mean={cf.mean():12.6f} cf_std={cf.std(ddof=0):10.6f} '
                  f'cf_min={cf.min():10.4f} cf_max={cf.max():10.4f}')
    print()
    print(f'CHECK RESULT: {"READY" if ok else "NOT READY"}')
    return 0 if ok else 4


def build_union_only(fa, ds, split, affinity):
    """Rebuild the M5 union exactly as the frozen driver does."""
    import build_m5_rq5 as b
    class _P:  # the union build only needs p5 for set_nk's namespace sync
        N_K = 5
    p5 = b._P5 if b._P5 is not None else _P()
    dumps, prov = b.load_scope_inputs(fa, ds, affinity, split, 'm5')
    if dumps is None:
        raise SystemExit(f'{ds}/{split}: missing input {prov}')
    b.set_nk(fa, p5, 'm5')
    return b.build_scope_union(fa, dumps, 'm5')


def _p5_stub():
    return None


# ===========================================================================
# cli
# ===========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Measure the M5 CFLA rows by wrapping the frozen build_m5_rq5 '
                    'driver (which is never modified).')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--affinity', default=bm.CANONICAL_AFFINITY)
    ap.add_argument('--tag', default='m5rq5cfla')
    ap.add_argument('--dry-run', action='store_true',
                    help='run the whole CPU pipeline and write nothing')
    ap.add_argument('--check', action='store_true',
                    help='verify the CFLA inputs and the union/parquet alignment, '
                         'then exit (writes nothing)')
    a = ap.parse_args(argv)
    a.datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    for d in a.datasets:
        if d not in DATASETS:
            ap.error(f'unknown dataset {d!r}')
    return a


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    STATE['datasets'] = list(a.datasets)

    # The historical import needs the frozen driver's own path setup.
    fa, p5 = bm.import_historical()
    bm._P5 = p5

    if a.check:
        return check_mode(a, fa)

    install_patches()
    avail = probe_availability()
    plan, agree, want = plan_supersede(a)
    sup = record_supersede(plan, want, a.dry_run)

    print('=' * 78)
    print('build_m5_cfla.py  (wrapper around the FROZEN build_m5_rq5.py)')
    print('=' * 78)
    print(f'  datasets        {a.datasets}')
    print(f'  cfla available  '
          + ', '.join(f'{d}={v["available"]}' for d, v in avail.items()))
    for d, v in avail.items():
        if not v['available']:
            print(f'    {d}: ' + '; '.join(v['reasons']))
    print(f'  patches         in-memory only: load_scope_inputs (hint), '
          f'evaluate_strategies, bootstrap_pairs, cfla_rows_blocked, '
          f'skipped_entries')
    print(f'  supersede       {[s["run"] for s in plan] if plan else "none"}')
    if sup:
        print(f'  supersede note  {sup["reason"][:300]}...')
    print()

    bm_argv = ['--datasets', ','.join(a.datasets), '--scopes', 'm5,spectral4',
               '--tag', a.tag, '--affinity', a.affinity]
    if a.dry_run:
        bm_argv.append('--dry-run')
    rc = bm.main(bm_argv)

    # A machine-readable record next to the run it describes.
    rec = {'wrapper': 'build_m5_cfla.py', 'frozen_driver': str(DRIVERS /
                                                               'build_m5_rq5.py'),
           'frozen_driver_sha256': bm.sha256_file(DRIVERS / 'build_m5_rq5.py'),
           'argv': _sys.argv, 'datasets': a.datasets,
           'cfla_available': avail, 'superseded_runs': plan,
           'supersede_written': bool(sup) and not a.dry_run,
           'dry_run': bool(a.dry_run), 'exit_code': rc,
           'elapsed_s': round(time.time() - t0, 1),
           'alignment': STATE['alignment'], 'events': STATE['log'],
           'premerge_guard': STATE.get('premerge'),
           'candidate_nll_dir': str(CAND), 'scalers': str(SCALERS_PATH),
           'wrote_only_under': str(OUT)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # A dry run writes NO run directory and no aggregate; its record goes into a
    # clearly separated `dryrun/` folder so it can never be mistaken for a
    # measurement.
    sub = 'dryrun' if a.dry_run else 'runs'
    p = M5DIR / 'cfla_wrapper' / sub / f'BUILD_M5_CFLA{"_DRYRUN" if a.dry_run else ""}' \
                                      f'.{stamp}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'\nwrapper record ({sub}): {p}')
    return rc


if __name__ == '__main__':
    _sys.exit(main())
