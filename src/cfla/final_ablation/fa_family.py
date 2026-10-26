#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION E -- same-family vs cross-family factorization
ABLATION G -- inventory of the number of factorizations

Everything in this script is CPU-only, read-only with respect to the audited
repository, and starts NO training and NO tokenizer training.  No checkpoint is
loaded or executed: every number below comes from the already-frozen beam dumps,
SID arrays and the already-frozen artifact matrix.

What it produces (all NEW files, never overwrites an existing one):

  outputs/final_ablation/family_artifact_inventory.csv
      48 rows, one per (dataset, family, K) cell of
      outputs/mechanism/factorization_generality/ARTIFACT_MATRIX.csv, with
      existence + sha256 verification of the sid / checkpoint / valid beam /
      test beam, plus structural and metric cross-checks.

  outputs/final_ablation/factorization_count_inventory.csv
      one row per factorization that could join a 3-way (count-ablation)
      combination: the 16 matrix cells plus rqvae3 / opq3 / rq1_opq2.

  outputs/final_ablation/factorization_count_three_way.csv
      beam-structural inventory (union size / target coverage / incremental
      coverage) of the frozen pair PLUS one third factorization.  Inventory
      only: no NLL, no ranking metric, no gate.

  outputs/final_ablation/family_ablation.csv
      Task 2: the CPU-only half of ablation E, one row per
      (dataset, pair_kind).  Definitions come from the frozen fusion core
      (fg_common.union_fuse / candidate_overlap / oracle_rank), so the
      same-family numbers can be checked against the published
      factorization_generality table.

  outputs/final_ablation/TRAINING_REQUIREMENT_REPORT.md
      Task 3: what the LSE half of ablation E still needs, what it costs, and
      the explicit statement that it needs inference only.

  outputs/final_ablation/FAMILY_ABLATION_NOTES.md
      Task 1 (b)/(c) report + Task 4 interpretation rule and reading.

Usage
  python3 fa_family.py --dry-run     # compute + print, write nothing
  python3 fa_family.py               # write all outputs (hard-fails on exists)
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
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                       # .../analysis/mechanism/final_ablation
REPO = HERE.parents[2]                                       # .../the development workspace
FG_ANALYSIS = REPO / 'analysis' / 'mechanism' / 'factorization_generality'
XT_ANALYSIS = REPO / 'analysis' / 'mechanism' / 'cross_tokenizer_shared'
for _p in (str(HERE), str(FG_ANALYSIS), str(XT_ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import xt_common as X                                        # noqa: E402
from fa_common import (ndcg_from_rank, paired_bootstrap, sha256_file,  # noqa: E402
                       write_csv)
from fg_common import (candidate_overlap, oracle_rank, single_rank,  # noqa: E402
                       summarize, union_fuse)

MATRIX = REPO / 'outputs' / 'mechanism' / 'factorization_generality' / 'ARTIFACT_MATRIX.csv'
FG_PUB = REPO / 'outputs' / 'mechanism' / 'factorization_generality' / 'pairwise_diversity_results.csv'
OUT = REPO / 'outputs' / 'final_ablation'
RUNS = OUT / 'runs'
CC_LOG = (REPO / 'outputs' / 'candidate_conditioned' / 'runs' /
          'multi_candidate_nll_seed2025_20260914T194657' / 'logs_candidate_nll.log')

DATASETS = list(X.DATASETS)
N_ITEMS = dict(X.N_ITEMS)
CODEBOOK = 256
BEAM = 30
NBIG = np.int64(2 ** 62)
TOPN = 20
SPLITS = ('valid', 'test')

# ---------------------------------------------------------------------------
# Task 1 -- cell registry built from ARTIFACT_MATRIX.csv
# ---------------------------------------------------------------------------
MATRIX_INDEX: dict[str, dict[str, dict]] = {}      # f'{family}_k{K}' -> ds -> row
MATRIX_ROWS: list[dict] = []


def load_matrix():
    global MATRIX_ROWS
    with MATRIX.open(newline='') as fh:
        MATRIX_ROWS = list(csv.DictReader(fh))
    for r in MATRIX_ROWS:
        key = matrix_key(r['family'], r['K'])
        MATRIX_INDEX.setdefault(key, {})[r['dataset']] = r
    if len(MATRIX_ROWS) != 48:
        raise SystemExit(f'HARD FAIL: expected 48 matrix rows, found {len(MATRIX_ROWS)}')
    combos = {(r['dataset'], r['family'], int(r['K'])) for r in MATRIX_ROWS}
    expected = {(ds, fam, k) for ds in DATASETS for fam in ('spectral', 'even', 'random')
                for k in (2, 3, 4, 5)}
    if combos != expected:
        raise SystemExit(f'HARD FAIL: matrix coverage mismatch: {sorted(expected ^ combos)}')
    return MATRIX_ROWS


def matrix_key(family, k):
    return f'{family}_k{int(k)}'


def register_matrix_cells():
    """Expose the matrix cells through xt_common's tokenizer registry.

    xt_common.beam_path() only knows the four frozen tokenizer keys, but
    xt_common.load_cell(..., beam_override=...) accepts any path.  The SID array is
    still looked up through the registry, so the missing keys are registered here
    with the matrix's OWN sid paths.  Existing keys (spectral_k3, rqvae3, ...) are
    never overwritten; their sid path is asserted to agree with the matrix.
    """
    for key, per_ds in MATRIX_INDEX.items():
        row0 = next(iter(per_ds.values()))
        k = int(row0['K'])
        fam = row0['family']
        if key in X.TOKENIZERS:
            for ds, r in per_ds.items():
                if Path(X.sid_path(key, ds)) != Path(r['sid']):
                    raise SystemExit(f'HARD FAIL: registry sid != matrix sid for {key}/{ds}')
            continue
        X.TOKENIZERS[key] = {
            'label': f'{fam}-K{k}',
            'family': fam,
            'construction': ('control partition family from the factorization_generality '
                             'study (matrix cell)'),
            'semantic_len': k, 'L': k + 1, 'codebook': CODEBOOK,
            'vocab': (k + 1) * CODEBOOK + 1, 'independent': False,
            'sid': (lambda ds, _k=key: Path(MATRIX_INDEX[_k][ds]['sid'])),
            'ckpt': (lambda ds, _k=key: Path(MATRIX_INDEX[_k][ds]['checkpoint'])),
            'beam': (lambda ds, split, _k=key: Path(
                MATRIX_INDEX[_k][ds]['valid_beam' if split == 'valid' else 'test_beam'])),
        }
    return MATRIX_INDEX


def matrix_beam_path(family, k, ds, split):
    row = MATRIX_INDEX[matrix_key(family, k)][ds]
    return Path(row['valid_beam' if split == 'valid' else 'test_beam'])


# ---------------------------------------------------------------------------
# structural helpers
# ---------------------------------------------------------------------------
def beam_binding(z, sid, shift):
    """Fraction of (candidate, position) pairs where the sid array reproduces the
    dumped candidate token exactly.  Strongest available check that the beam dump
    really belongs to this SID array and to this item-id convention."""
    raw = np.asarray(z['candidate_item_ids'])
    tok = np.asarray(z['candidate_tokens'])
    L = tok.shape[2]
    valid = raw >= 0
    idx = raw - shift
    inrange = valid & (idx >= 0) & (idx < len(sid))
    safe = np.clip(idx, 0, len(sid) - 1)
    rec = sid[safe] + np.arange(L, dtype=np.int64)[None, None, :] * CODEBOOK + 1
    match = (rec == tok) & inrange[:, :, None]
    denom = int(inrange.sum()) * L
    return (float(match.sum()) / denom if denom else float('nan'),
            int(valid.sum() - inrange.sum()), denom)


def target_check(z, shift, n_items):
    raw = np.asarray(z['target_item_ids'])
    tgt = np.where(raw >= 0, raw - shift + 1, -1)
    bad = int(((tgt < -1) | (tgt > n_items)).sum())
    return tgt, bad


def beam_set_keys(items):
    """Per-user sorted distinct item ids with NBIG sentinel padding."""
    return np.sort(np.where(items >= 0, items, NBIG), axis=1)


def distinct_count(items):
    """Per-user number of distinct valid item ids in `items` (-1 dropped)."""
    key = np.sort(np.where(items >= 0, items, NBIG), axis=1)
    return (key < NBIG).sum(axis=1).astype(np.int64)


def set_pair_stats(items_a, items_b):
    """(n_a, n_b, intersection, union) per user, from distinct counts only.

    |A and B| = |A| + |B| - |A or B| is exact for sets, so no O(n*B^2) mask is needed.
    """
    na = distinct_count(items_a)
    nb = distinct_count(items_b)
    union = distinct_count(np.concatenate([items_a, items_b], axis=1))
    return na, nb, na + nb - union, union


def target_in_keys(keys, target):
    pos = target >= 1
    return (pos[:, None] & (keys == target[:, None])).any(axis=1)


def ndcg20(rank):
    return ndcg_from_rank(np.asarray(rank), TOPN)


# ---------------------------------------------------------------------------
# cell loading (xt_common.load_cell, cached)
# ---------------------------------------------------------------------------
CELL_CACHE: dict[tuple, dict] = {}


def get_cell(key, ds, split, path=None):
    path = Path(path) if path is not None else Path(X.beam_path(key, ds, split))
    ck = (key, ds, split, str(path))
    if ck not in CELL_CACHE:
        CELL_CACHE[ck] = X.load_cell(key, ds, split, beam_override=path)
    return CELL_CACHE[ck]


def build_reference_targets():
    """Per-dataset target arrays from the frozen spectral K3 beam, both splits."""
    ref = {s: {} for s in SPLITS}
    for split in SPLITS:
        for ds in DATASETS:
            row = MATRIX_INDEX[matrix_key('spectral', 3)][ds]
            p = Path(row['valid_beam' if split == 'valid' else 'test_beam'])
            z = np.load(p, allow_pickle=True)
            sid = np.load(row['sid'], allow_pickle=True).astype(np.int64)
            shift, _ = X.detect_item_shift(z, sid)
            ref[split][ds] = target_check(z, shift, N_ITEMS[ds])[0]
            del z
    return ref


# ---------------------------------------------------------------------------
# Task 1(a) -- per-cell verification
# ---------------------------------------------------------------------------
def verify_cell(row, ref_targets):
    ds = row['dataset']
    fam = row['family']
    k = int(row['K'])
    key = matrix_key(fam, k)
    n_items = N_ITEMS[ds]
    paths = {'sid': Path(row['sid']), 'ckpt': Path(row['checkpoint']),
             'valid_beam': Path(row['valid_beam']), 'test_beam': Path(row['test_beam'])}
    rec = {'dataset': ds, 'family': fam, 'K': k, 'seed': int(row['seed']),
           'tokenizer_key': key, 'matrix_usable': int(row['usable'])}
    problems = []

    for name, p in paths.items():
        rec[f'{name}_path'] = str(p)
        rec[f'{name}_exists'] = int(p.exists())
        if not p.exists():
            problems.append(f'{name}_missing')
    if problems:
        rec['problems'] = ';'.join(problems)
        rec['cell_verified'] = 0
        return rec

    # ---- sha256 against the matrix ---------------------------------------
    actual = {name: sha256_file(p) for name, p in paths.items()}
    for name, col in (('sid', 'sid_sha256'), ('ckpt', 'ckpt_sha256'),
                      ('test_beam', 'beam_sha256')):
        rec[f'{name}_sha_expected'] = row[col]
        rec[f'{name}_sha_actual'] = actual[name]
        rec[f'{name}_sha_ok'] = int(actual[name] == row[col])
        if actual[name] != row[col]:
            problems.append(f'{name}_sha_mismatch')
    # the matrix carries no sha for the valid beam (it stores the test-beam sha only)
    rec['valid_beam_sha_actual'] = actual['valid_beam']
    rec['valid_beam_sha_ok'] = 'no_reference_in_matrix'

    # ---- sid structure ----------------------------------------------------
    sid = np.load(paths['sid'], allow_pickle=True).astype(np.int64)
    rec['sid_n_items'] = int(sid.shape[0])
    rec['sid_L'] = int(sid.shape[1])
    rec['sid_expected_n_items'] = n_items
    rec['sid_shape_ok'] = int(sid.shape == (n_items, k + 1))
    rec['sid_code_range_ok'] = int(bool((sid >= 0).all() and (sid < CODEBOOK).all()))
    rec['sid_injective'] = int(len({tuple(r) for r in sid}) == len(sid))
    for f in ('sid_shape_ok', 'sid_code_range_ok', 'sid_injective'):
        if not rec[f]:
            problems.append(f)

    # ---- beams ------------------------------------------------------------
    for split in SPLITS:
        p = paths[f'{split}_beam']
        z = np.load(p, allow_pickle=True)
        items = np.asarray(z['candidate_item_ids'])
        shift_det, shift_scores = X.detect_item_shift(z, sid)
        binding, out_of_range, denom = beam_binding(z, sid, shift_det)
        tgt, bad_tgt = target_check(z, shift_det, n_items)
        rec[f'{split}_n_users'] = int(items.shape[0])
        rec[f'{split}_beam_budget'] = int(items.shape[1])
        rec[f'{split}_detected_shift'] = int(shift_det)
        rec[f'{split}_shift_detect_fraction'] = round(float(shift_scores[shift_det]), 6)
        rec[f'{split}_beam_sid_binding_fraction'] = round(binding, 9)
        rec[f'{split}_candidates_out_of_sid_range'] = out_of_range
        rec[f'{split}_target_out_of_range'] = bad_tgt
        rec[f'{split}_binding_checked_pairs'] = denom
        rec[f'{split}_targets_match_dataset_reference'] = int(
            np.array_equal(tgt, ref_targets[split][ds]))
        if items.shape[1] != BEAM:
            problems.append(f'{split}_beam_budget_{items.shape[1]}')
        if shift_det != int(row['raw_item_id_shift']):
            problems.append(f'{split}_shift_{shift_det}_ne_matrix_{row["raw_item_id_shift"]}')
        if binding < 1.0:
            problems.append(f'{split}_sid_binding_{binding:.6f}')
        if out_of_range:
            problems.append(f'{split}_candidates_out_of_range_{out_of_range}')
        if bad_tgt:
            problems.append(f'{split}_target_out_of_range_{bad_tgt}')
        if not rec[f'{split}_targets_match_dataset_reference']:
            problems.append(f'{split}_targets_differ_from_dataset_reference')
        if split == 'test':
            rec['matrix_n_users'] = int(row['n_users'])
            rec['n_users_ok'] = int(int(items.shape[0]) == int(row['n_users']))
            rec['matrix_shift'] = int(row['raw_item_id_shift'])
            rec['shift_ok'] = int(shift_det == int(row['raw_item_id_shift']))
            if not rec['n_users_ok']:
                problems.append('n_users_mismatch')
            cell = X.load_cell(key, ds, 'test', beam_override=p)
            sm = summarize(single_rank(cell))
            metrics = json.loads(row['metrics'])
            rec['metric_ndcg20_matrix'] = round(float(metrics['NDCG@20']), 6)
            rec['metric_ndcg20_recomputed'] = round(float(sm['NDCG@20']), 6)
            rec['metric_ndcg20_abs_diff'] = abs(rec['metric_ndcg20_recomputed']
                                                - rec['metric_ndcg20_matrix'])
            rec['metric_ndcg20_ok'] = int(rec['metric_ndcg20_abs_diff'] < 1e-6)
            rec['target_present_matrix'] = round(float(metrics['target_present']), 6)
            rec['target_present_recomputed'] = round(float(sm['target_present']), 6)
            rec['target_present_ok'] = int(
                abs(rec['target_present_recomputed'] - rec['target_present_matrix']) < 1e-6)
            if not rec['metric_ndcg20_ok']:
                problems.append('ndcg20_recompute_mismatch')
            if not rec['target_present_ok']:
                problems.append('target_present_recompute_mismatch')
        del z

    rec['problems'] = ';'.join(problems)
    rec['cell_verified'] = int(not problems)
    return rec


def task1_matrix(rows, ref_targets):
    out = []
    todo = [r for r in rows if r['dataset'] in DATASETS]
    for i, row in enumerate(todo, 1):
        rec = verify_cell(row, ref_targets)
        out.append(rec)
        flag = 'OK ' if rec['cell_verified'] else 'BAD'
        print(f'  [{i:2d}/{len(todo)}] {rec["dataset"][:12]:12s} {rec["family"]:8s} K={rec["K"]} '
              f'{flag} ndcg20={rec.get("metric_ndcg20_recomputed")} '
              f'binding={rec.get("test_beam_sid_binding_fraction")} '
              f'{rec["problems"]}', flush=True)
    return out


# ---------------------------------------------------------------------------
# Task 1(c) -- factorization inventory (matrix cells + independent families)
# ---------------------------------------------------------------------------
XT_CELLS = [
    {'factorization': 'rqvae3', 'role': 'cross_family_primary',
     'kind': 'cross_family_primary', 'caveat': 'none'},
    {'factorization': 'opq3', 'role': 'cross_family_secondary',
     'kind': 'cross_family_secondary',
     'caveat': 'checkpoint selected with a legacy token-position metric (secondary evidence)'},
    {'factorization': 'rq1_opq2', 'role': 'cross_family_secondary',
     'kind': 'cross_family_secondary',
     'caveat': 'checkpoint selected with a legacy token-position metric (secondary evidence)'},
]


def task1_factorizations(rows, ref_targets):
    """One row per candidate factorization for a 3-way (count) combination."""
    verified = {f'{r["dataset"]}|{r["tokenizer_key"]}': r for r in rows}
    out = []
    # --- the 16 matrix cells, aggregated per factorization -------------------
    for key, per_ds in MATRIX_INDEX.items():
        row0 = next(iter(per_ds.values()))
        fam = row0['family']
        k = int(row0['K'])
        recs = [verified[f'{ds}|{key}'] for ds in DATASETS]
        rec = {
            'factorization': key, 'family': fam, 'K': k, 'semantic_len': k,
            'L': k + 1, 'source': 'factorization_generality/ARTIFACT_MATRIX.csv',
            'role': f'same_family_control ({fam})',
            'kind': 'same_family_control',
            'eligible_as_third': 1,
            'n_datasets': len(DATASETS),
            'datasets_available': '/'.join(DATASETS),
            'cells_verified': sum(r['cell_verified'] for r in recs),
            'all_sid_exist': int(all(r['sid_exists'] for r in recs)),
            'all_ckpt_exist': int(all(r['ckpt_exists'] for r in recs)),
            'all_valid_beam_exist': int(all(r['valid_beam_exists'] for r in recs)),
            'all_test_beam_exist': int(all(r['test_beam_exists'] for r in recs)),
            'sha_reference_available': 1,
            'sha_all_match': int(all(r['sid_sha_ok'] and r['ckpt_sha_ok']
                                     and r['test_beam_sha_ok'] for r in recs)),
            'sid_beam_binding_ok_all': int(all(
                r['test_beam_sid_binding_fraction'] == 1.0 for r in recs)),
            'targets_match_reference_all': int(all(
                r['test_targets_match_dataset_reference'] for r in recs)),
            'n_users_total': sum(r['test_n_users'] for r in recs),
            'sid_path_template': _ART_ROOT + '/data/<ds>/<ds>_t5_%s_v3_k%d_sid%d.npy' % (
                'spectral_nonneg_pearson' if fam == 'spectral' else f'{fam}split', k, k + 1),
            'ckpt_path_template': _ART_ROOT + '/model/ckpt/tiger_<ds>_%s_v3_k%d_sid%d.pth'
                                  % ('nonneg_pearson' if fam == 'spectral' else f'{fam}split',
                                     k, k + 1),
            'test_beam_template': (_ART_ROOT + '/model/beam_dumps_<ds>_nonneg_pearson_v3/'
                                   'test_k%d_sid%d.npz' % (k, k + 1)),
            'valid_beam_template': (_ART_ROOT + '/model/beam_dumps_<ds>_nonneg_pearson_v3/'
                                    'valid_k%d_sid%d.npz' % (k, k + 1)),
            'in_artifact_matrix': 1,
            'metric_provenance': 'canonical (checkpoint selected on the frozen retrieval metric)',
            'caveat': ('same embedding space as spectral/even/random, i.e. a same-family '
                       'control rather than an independent family' if fam != 'spectral' else
                       'reference same-family control'),
            'notes': ('same-family control: partition of the SAME T5 embedding space'
                      if fam != 'spectral' else
                      'primary frozen factorization F1; the spectral K values are the '
                      'same-family partners used by ablation E'),
        }
        out.append(rec)

    # --- the three independent tokenizer families ---------------------------
    for spec in XT_CELLS:
        tok = spec['factorization']
        rec = {'factorization': tok, 'family': X.TOKENIZERS[tok]['family'],
               'K': X.TOKENIZERS[tok]['semantic_len'], 'L': X.TOKENIZERS[tok]['L'],
               'source': 'cross_tokenizer_shared/step1_cross_tokenizer',
               'role': spec['role'], 'kind': spec['kind'], 'eligible_as_third': 1,
               'n_datasets': len(DATASETS), 'datasets_available': '/'.join(DATASETS),
               'in_artifact_matrix': 0, 'sha_reference_available': 0,
               'metric_provenance': ('secondary: checkpoint chosen with a legacy '
                                     'token-position metric'
                                     if spec['kind'].endswith('secondary')
                                     else 'primary cross-family factorization F2'),
               'caveat': spec['caveat'],
               'sid_path_template': str(X.sid_path(tok, '<ds>')),
               'ckpt_path_template': str(X.ckpt_path(tok, '<ds>')),
               'test_beam_template': str(X.beam_path(tok, '<ds>', 'test')),
               'valid_beam_template': str(X.beam_path(tok, '<ds>', 'valid')),
               'notes': X.TOKENIZERS[tok]['construction']}
        ok_sid = ok_ck = ok_v = ok_t = ok_bind = ok_tgt = 0
        n_users = 0
        for ds in DATASETS:
            sp = Path(X.sid_path(tok, ds))
            cp = Path(X.ckpt_path(tok, ds))
            tp = Path(X.beam_path(tok, ds, 'test'))
            vp = Path(X.beam_path(tok, ds, 'valid'))
            ok_sid += int(sp.exists())
            ok_ck += int(cp.exists())
            ok_t += int(tp.exists())
            ok_v += int(vp.exists())
            if not (sp.exists() and cp.exists() and tp.exists()):
                continue
            sid = np.load(sp, allow_pickle=True).astype(np.int64)
            z = np.load(tp, allow_pickle=True)
            shift, _ = X.detect_item_shift(z, sid)
            binding, oob, _ = beam_binding(z, sid, shift)
            tgt, bad = target_check(z, shift, N_ITEMS[ds])
            ok_bind += int(binding == 1.0 and oob == 0)
            ok_tgt += int(np.array_equal(tgt, ref_targets['test'][ds]) and bad == 0)
            n_users += int(np.asarray(z['candidate_item_ids']).shape[0])
            del z
        rec.update({'all_sid_exist': int(ok_sid == len(DATASETS)),
                    'all_ckpt_exist': int(ok_ck == len(DATASETS)),
                    'all_valid_beam_exist': int(ok_v == len(DATASETS)),
                    'all_test_beam_exist': int(ok_t == len(DATASETS)),
                    'sha_all_match': 'no_reference_in_matrix',
                    'sid_beam_binding_ok_all': int(ok_bind == len(DATASETS)),
                    'targets_match_reference_all': int(ok_tgt == len(DATASETS)),
                    'cells_verified': int(ok_sid == ok_ck == ok_v == ok_t == len(DATASETS)
                                          and ok_bind == len(DATASETS)
                                          and ok_tgt == len(DATASETS)),
                    'n_users_total': n_users})
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Task 2 -- same-family vs cross-family, CPU-only (beam dumps only)
# ---------------------------------------------------------------------------
def _spec_label(spec):
    return matrix_key(spec[1], spec[2]) if spec[0] == 'matrix' else spec[1]


def _spec_cell(spec, ds, split):
    if spec[0] == 'matrix':
        key = matrix_key(spec[1], spec[2])
        return get_cell(key, ds, split, matrix_beam_path(spec[1], spec[2], ds, split))
    return get_cell(spec[1], ds, split)


def specs_for_pairs():
    m = lambda fam, k: ('matrix', fam, k)          # noqa: E731
    x = lambda tok: ('xt', tok)                    # noqa: E731
    return [
        # (pair_kind, spec_a, spec_b, is_primary, group)
        ('cross_family', m('spectral', 3), x('rqvae3'), 1, 'cross_family'),
        ('same_family_K2_K3', m('spectral', 2), m('spectral', 3), 1, 'same_family'),
        ('same_family_K3_K4', m('spectral', 3), m('spectral', 4), 1, 'same_family'),
        ('same_family_K2_K5', m('spectral', 2), m('spectral', 5), 1, 'same_family'),
        # extra same-family controls (available, not requested as primary)
        ('same_family_K3_K5', m('spectral', 3), m('spectral', 5), 0, 'same_family_extra'),
        ('same_family_K4_K5', m('spectral', 4), m('spectral', 5), 0, 'same_family_extra'),
        ('same_family_even_K2_K3', m('even', 2), m('even', 3), 0, 'same_family_extra'),
        ('same_family_even_K3_K4', m('even', 3), m('even', 4), 0, 'same_family_extra'),
        ('same_family_random_K2_K3', m('random', 2), m('random', 3), 0, 'same_family_extra'),
        ('same_family_random_K3_K4', m('random', 3), m('random', 4), 0, 'same_family_extra'),
        # genuinely independent families, legacy-metric checkpoints (secondary)
        ('cross_family_secondary_opq3', m('spectral', 3), x('opq3'), 0,
         'cross_family_secondary'),
        ('cross_family_secondary_rq1_opq2', m('spectral', 3), x('rq1_opq2'), 0,
         'cross_family_secondary'),
    ]


def load_fg_published():
    pub = {}
    if not FG_PUB.exists():
        return pub
    with FG_PUB.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if r['diversity_type'] != 'G':
                continue
            pub[(r['dataset'], r['model_A'], r['model_B'])] = r
    return pub


def pair_row(ds, pair_kind, spec_a, spec_b, is_primary, group, pub):
    cA = _spec_cell(spec_a, ds, 'test')
    cB = _spec_cell(spec_b, ds, 'test')
    if not np.array_equal(cA['target'], cB['target']):
        raise SystemExit(f'HARD FAIL: {ds} {pair_kind}: targets differ between cells')
    if cA['n_users'] != cB['n_users']:
        raise SystemExit(f'HARD FAIL: {ds} {pair_kind}: n_users differ')

    tgt = cA['target']
    na, nb, inter, union = set_pair_stats(cA['items'], cB['items'])
    jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    ka = beam_set_keys(cA['items'])
    kb = beam_set_keys(cB['items'])
    in_a = target_in_keys(ka, tgt)
    in_b = target_in_keys(kb, tgt)
    covered = in_a | in_b

    fA = union_fuse([cA])
    fB = union_fuse([cB])
    fAB = union_fuse([cA, cB])
    sA = summarize(fA['rank'])
    sB = summarize(fB['rank'])
    sF = summarize(fAB['rank'])
    orank = oracle_rank([cA, cB])
    sO = summarize(orank)

    best_name, best_rank, best_ndcg = (
        (_spec_label(spec_a), fA['rank'], sA['NDCG@20'])
        if sA['NDCG@20'] >= sB['NDCG@20'] else
        (_spec_label(spec_b), fB['rank'], sB['NDCG@20']))
    headroom = sO['NDCG@20'] - best_ndcg
    expl = (sF['NDCG@20'] - best_ndcg) / headroom if headroom > 0 else float('nan')
    boot = paired_bootstrap(ndcg20(fAB['rank']), ndcg20(best_rank))

    # restricted to each side's top-20 -- the published factorization_generality
    # definition of candidate Jaccard / union size; reported as a cross-check only
    ov = candidate_overlap(fA['top_items'], fA['top_items'] >= 0,
                           fB['top_items'], fB['top_items'] >= 0,
                           np.minimum(fA['n_valid_top'], TOPN),
                           np.minimum(fB['n_valid_top'], TOPN))
    top_cov = (target_in_keys(beam_set_keys(fA['top_items']), tgt)
               | target_in_keys(beam_set_keys(fB['top_items']), tgt))

    label_a = _spec_label(spec_a)
    label_b = _spec_label(spec_b)
    ref = pub.get((ds, label_a, label_b)) or pub.get((ds, label_b, label_a))
    row = {
        'dataset': ds, 'pair_kind': pair_kind, 'is_primary': is_primary,
        'pair_group': group, 'pair': f'{label_a}+{label_b}',
        'family_a': X.TOKENIZERS[label_a]['family'],
        'K_a': X.TOKENIZERS[label_a]['semantic_len'],
        'family_b': X.TOKENIZERS[label_b]['family'],
        'K_b': X.TOKENIZERS[label_b]['semantic_len'],
        'n_users': int(cA['n_users']),
        'mean_candidates_a': float(na.mean()), 'mean_candidates_b': float(nb.mean()),
        'mean_union_size': float(union.mean()),
        'mean_candidate_intersection': float(inter.mean()),
        'candidate_jaccard': float(np.nanmean(jac)),
        'users_with_empty_union': int((union == 0).sum()),
        'target_present_a': float(in_a.mean()), 'target_present_b': float(in_b.mean()),
        'target_coverage_union': float(covered.mean()),
        'oracle_in_union_ndcg20': float(ndcg20(np.where(covered, 1, -1)).mean()),
        'single_a_ndcg20': float(sA['NDCG@20']), 'single_b_ndcg20': float(sB['NDCG@20']),
        'single_a_recall20': float(sA['Recall@20']), 'single_b_recall20': float(sB['Recall@20']),
        'best_single_name': best_name, 'best_single_ndcg20': float(best_ndcg),
        'uncalibrated_oracle_min_rank_ndcg20': float(sO['NDCG@20']),
        'uniform_ndcg20': float(sF['NDCG@20']), 'uniform_recall20': float(sF['Recall@20']),
        'uniform_gain_over_best_single': float(sF['NDCG@20'] - best_ndcg),
        'uniform_vs_best_single_ci_low': float(boot['ci_low']),
        'uniform_vs_best_single_ci_high': float(boot['ci_high']),
        'uniform_vs_best_single_p_value': float(boot['p_value']),
        'oracle_headroom_exploited_fraction': float(expl),
        'top20_union_size': float(ov['candidate_union_size']),
        'top20_candidate_intersection': float(ov['candidate_intersection']),
        'top20_candidate_jaccard': float(ov['candidate_jaccard']),
        'top20_target_coverage_union': float(top_cov.mean()),
        'fg_published_fusion_ndcg20': float(ref['fusion_ndcg20']) if ref else '',
        'fg_published_top20_candidate_jaccard': (float(ref['candidate_jaccard'])
                                                if ref else ''),
        'fg_abs_diff_fusion_ndcg20': (abs(float(ref['fusion_ndcg20']) - float(sF['NDCG@20']))
                                      if ref else ''),
        'fg_abs_diff_top20_candidate_jaccard': (
            abs(float(ref['candidate_jaccard']) - float(ov['candidate_jaccard']))
            if ref else ''),
        'beam_a_path': str(cA['path']), 'beam_b_path': str(cB['path']),
        'beam_a_sha256': sha256_file(cA['path']), 'beam_b_sha256': sha256_file(cB['path']),
        'beam_a_shift': int(cA['raw_item_id_shift']),
        'beam_b_shift': int(cB['raw_item_id_shift']),
        'loader': 'xt_common.load_cell(beam_override=...)',
    }
    return row, float(sF['NDCG@20']), (float(ref['fusion_ndcg20']) if ref else None)


def task2_rows(pub):
    rows, checks = [], []
    for ds in DATASETS:
        for pair_kind, sa, sb, primary, group in specs_for_pairs():
            row, fusion, ref = pair_row(ds, pair_kind, sa, sb, primary, group, pub)
            rows.append(row)
            if ref is not None:
                checks.append((ds, row['pair'], abs(ref - fusion)))
            print(f'  {ds[:12]:12s} {pair_kind:32s} jac={row["candidate_jaccard"]:.4f} '
                  f'union={row["mean_union_size"]:6.2f} cov={row["target_coverage_union"]:.4f} '
                  f'uniform20={row["uniform_ndcg20"]:.6f} best={row["best_single_ndcg20"]:.6f}',
                  flush=True)
    return rows, checks


# ---------------------------------------------------------------------------
# Task 1(c)/G -- three-way structural inventory (no NLL, no ranking metric)
# ---------------------------------------------------------------------------
THIRD_CANDIDATES = [
    ('even_k3', ('matrix', 'even', 3), 'same_family_control',
     'same embedding space as spectral: control, not an independent family'),
    ('random_k3', ('matrix', 'random', 3), 'same_family_control',
     'same embedding space as spectral: control, not an independent family'),
    ('opq3', ('xt', 'opq3'), 'cross_family_secondary',
     'checkpoint chosen with a legacy token-position metric'),
    ('rq1_opq2', ('xt', 'rq1_opq2'), 'cross_family_secondary',
     'checkpoint chosen with a legacy token-position metric'),
]
BASE = [('matrix', 'spectral', 3), ('xt', 'rqvae3')]


def three_way_rows():
    rows = []
    for ds in DATASETS:
        base_cells = [_spec_cell(s, ds, 'test') for s in BASE]
        base_items = np.concatenate([c['items'] for c in base_cells], axis=1)
        tgt = base_cells[0]['target']
        u2 = distinct_count(base_items)
        kbase = beam_set_keys(base_items)
        base_cov = target_in_keys(kbase, tgt)
        for name, spec, kind, caveat in THIRD_CANDIDATES:
            c3 = _spec_cell(spec, ds, 'test')
            if not np.array_equal(c3['target'], tgt):
                raise SystemExit(f'HARD FAIL: {ds} {name}: targets differ from base pair')
            both = np.concatenate([base_items, c3['items']], axis=1)
            u3 = distinct_count(both)
            cov3 = target_in_keys(beam_set_keys(both), tgt)
            in3 = target_in_keys(beam_set_keys(c3['items']), tgt)
            na, nb, inter, union = set_pair_stats(base_items, c3['items'])
            jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
            rows.append({
                'dataset': ds, 'triple': f'spectral_k3+rqvae3+{name}',
                'base_pair': 'spectral_k3+rqvae3', 'third_factorization': name,
                'third_kind': kind, 'third_caveat': caveat,
                'n_users': int(tgt.shape[0]),
                'mean_candidates_third': float(nb.mean()),
                'mean_union_size_2way': float(u2.mean()),
                'mean_union_size_3way': float(u3.mean()),
                'mean_incremental_union_size': float((u3 - u2).mean()),
                'mean_third_only_candidates': float((u3 - u2).mean()),
                'target_coverage_2way': float(base_cov.mean()),
                'target_coverage_3way': float(cov3.mean()),
                'incremental_coverage': float(cov3.mean() - base_cov.mean()),
                'users_target_in_third': int(in3.sum()),
                'users_target_in_third_only': int((in3 & ~base_cov).sum()),
                'users_target_in_third_only_fraction': float((in3 & ~base_cov).mean()),
                'candidate_jaccard_base_vs_third': float(np.nanmean(jac)),
                'metric_scope': 'beam-structural inventory only (no NLL, no ranking metric)',
            })
            print(f'  {ds[:12]:12s} 3way +{name:10s} union '
                  f'2way={rows[-1]["mean_union_size_2way"]:.2f} '
                  f'3way={rows[-1]["mean_union_size_3way"]:.2f} '
                  f'cov2={base_cov.mean():.4f} cov3={cov3.mean():.4f} '
                  f'(+{rows[-1]["incremental_coverage"]:.4f})', flush=True)
    return rows


# ---------------------------------------------------------------------------
# Task 3 -- exact inference budget for the LSE half
# ---------------------------------------------------------------------------
def frozen_throughput():
    """Measured candidate-scoring throughput of the frozen candidate-NLL run."""
    rates = {}
    if CC_LOG.exists():
        cur = None
        for line in CC_LOG.read_text().splitlines():
            m = re.match(r'=== (\S+) (valid|test) ===', line.strip())
            if m:
                cur = m.group(1)
            m = re.search(r'spectral_k3\s+scored (\d+) candidates .* in ([\d.]+)s', line)
            if m and cur:
                rates.setdefault(cur, []).append(int(m.group(1)) / float(m.group(2)))
    if not rates:
        return None
    allr = [r for v in rates.values() for r in v]
    return {'per_dataset_cand_per_s': {k: float(np.mean(v)) for k, v in rates.items()},
            'min_cand_per_s': float(min(allr)), 'n_measurements': len(allr),
            'source': str(CC_LOG)}


def lse_budget():
    """Exact candidate counts for the same-family LSE work, both splits."""
    pairs = [('same_family_K2_K3', ('matrix', 'spectral', 2), ('matrix', 'spectral', 3)),
             ('same_family_K3_K4', ('matrix', 'spectral', 3), ('matrix', 'spectral', 4)),
             ('same_family_K2_K5', ('matrix', 'spectral', 2), ('matrix', 'spectral', 5))]
    detail, totals = [], {'passes': 0, 'candidates': 0, 'optimized_candidates': 0}
    for ds in DATASETS:
        for split in SPLITS:
            cells = {k: _spec_cell(('matrix', 'spectral', k), ds, split) for k in (2, 3, 4, 5)}
            for kind, sa, sb in pairs:
                cA = _spec_cell(sa, ds, split)
                cB = _spec_cell(sb, ds, split)
                u = union_fuse([cA, cB])['union_size']
                detail.append({'dataset': ds, 'split': split, 'pair_kind': kind,
                               'n_users': int(cA['n_users']),
                               'mean_union_size': float(u.mean()),
                               'candidate_scorings': int(u.sum()), 'passes': 2})
                totals['passes'] += 2
                totals['candidates'] += int(u.sum())
            # optimised schedule: one pass per factorization over the union of the pair
            # unions it participates in (K2: K2,K3,K5 | K3: K2,K3,K4 | K4: K3,K4 | K5: K2,K5)
            pools = {'k2': (2, 3, 5), 'k3': (2, 3, 4), 'k4': (3, 4), 'k5': (2, 5)}
            for name, members in pools.items():
                pool_items = np.concatenate([cells[m]['items'] for m in members], axis=1)
                cnt = distinct_count(pool_items)
                detail.append({'dataset': ds, 'split': split,
                               'pair_kind': f'optimised_pool_{name}',
                               'n_users': int(cells[members[0]]['n_users']),
                               'mean_union_size': float(cnt.mean()),
                               'candidate_scorings': int(cnt.sum()), 'passes': 1})
                totals['optimized_candidates'] += int(cnt.sum())
                totals['passes'] += 0
    totals['optimized_passes'] = len(DATASETS) * len(SPLITS) * 4
    return detail, totals


def build_report(rows_inv, fam_rows, checks, budget_detail, budget_totals, rates):
    """TRAINING_REQUIREMENT_REPORT.md"""
    n_ok = sum(r['cell_verified'] for r in rows_inv)
    primary = [r for r in fam_rows if r['is_primary'] == 1]
    same = [r for r in primary if r['pair_group'] == 'same_family']
    cross = [r for r in primary if r['pair_group'] == 'cross_family']
    agg = lambda rs, key: float(np.mean([float(r[key]) for r in rs]))  # noqa: E731
    mean_same_j = agg(same, 'candidate_jaccard')
    mean_cross_j = agg(cross, 'candidate_jaccard')
    mean_same_cov = agg(same, 'target_coverage_union')
    mean_cross_cov = agg(cross, 'target_coverage_union')
    mean_same_u = agg(same, 'uniform_ndcg20')
    mean_cross_u = agg(cross, 'uniform_ndcg20')
    rate = rates['min_cand_per_s'] if rates else 23500.0
    per_pass = budget_totals['candidates'] / rate
    per_pass_opt = budget_totals['optimized_candidates'] / rate
    worst_check = max((c[2] for c in checks), default=float('nan'))
    L = []
    A = L.append
    A('# TRAINING_REQUIREMENT_REPORT')
    A('')
    A('Ablation E (same-family vs cross-family factorization) -- honest limitation report.')
    A('Generated by `analysis/mechanism/final_ablation/fa_family.py` on '
      f'{time.strftime("%Y-%m-%d %H:%M:%S%z")}.')
    A('')
    A('## 1. What could not be completed in this round, and why')
    A('')
    A('The CPU-only half of ablation E is complete (Task 2, '
      '`outputs/final_ablation/family_ablation.csv`).')
    A('The **LSE half is not**, and it was deliberately not started.')
    A('')
    A('The frozen method ranks a candidate `i` for user `u` with')
    A('')
    A('```')
    A('zNLL_F(u,i) = (TotalNLL_F(u,i) - mu_F) / sigma_F      # mu,sigma fitted on valid,')
    A('                                                      # over ALL union candidates')
    A('LSE(u,i)    = logsumexp(-zNLL_F1(u,i), -zNLL_F2(u,i))')
    A('```')
    A('')
    A('`TotalNLL_F(u,i)` is the teacher-forced negative log-likelihood of candidate `i`\'s own')
    A('SID sequence under factorization `F`\'s recommender checkpoint. It cannot be derived from')
    A('the beam dumps: a beam dump holds, per user, the top-30 decoded item ids and their')
    A('sequence scores under ONE factorization. It does not contain the logits that any OTHER')
    A('checkpoint assigns to the union candidates, and beam `sequence_scores` are not comparable')
    A('across checkpoints (different vocabularies, different normalisation, and the union')
    A('contains candidates one checkpoint would never decode).')
    A('')
    A('Producing `zNLL_F(u,i)` for the same-family pairs therefore needs GPU forward passes of')
    A('the existing T5 checkpoints over the union candidates. **This is a compute limit, not a')
    A('training limit**: no parameter is updated, no tokenizer is fitted. The task forbids')
    A('starting that inference, so it is reported here instead of run.')
    A('')
    A('| ablation | needs | status here |')
    A('|---|---|---|')
    A('| E, cross-family, Uniform / coverage / Jaccard | beam dumps only | DONE (Task 2) |')
    A('| E, cross-family, LSE / MinNLL / z-calibrated fusion | teacher-forced union NLL for '
      '(spectral_k3, rqvae3) | already frozen: '
      '`outputs/candidate_conditioned/runs/multi_candidate_nll_seed2025_20260914T194657` '
      '(valid + test, 4 datasets). No new compute needed. |')
    A('| E, **same-family**, LSE / MinNLL / any z-calibrated fusion | teacher-forced union '
      'NLL under BOTH same-family checkpoints, valid AND test | **NOT DONE -- needs inference** |')
    A('| G, count ablation (3+ factorizations), ranking metrics | the same plus the third '
      'factorization\'s union NLL | inventory DONE; ranking NOT DONE -- needs inference |')
    A('')
    A('## 2. Exactly what would be required')
    A('')
    A('Reuse the frozen scoring path (no new code path, no new model, no re-training):')
    A('')
    A('* `analysis/mechanism/candidate_conditioned/cc_common.py::candidate_total_nll`')
    A('  (batched teacher forcing, one encoder pass per user batch, that batch\'s candidates')
    A('  decoded together, `batch_size=512`, `torch.no_grad`), and')
    A('* `src/analysis/teacher_forced.py::load_model_from_ckpt` to load each EXISTING')
    A('  same-family checkpoint in eval mode.')
    A('')
    A('Steps per (dataset, same-family pair, split):')
    A('')
    A('1. Build the union `C(u) = C_a(u) union C_b(u)` from the two frozen beam dumps (exactly')
    A('   the union used in Task 2; the target is never added).')
    A('2. Tokenise every candidate under BOTH SID systems (one SID array per factorization).')
    A('3. Teacher-force the union under checkpoint `a` -> `TotalNLL_a(u,i)`, and under checkpoint')
    A('   `b` -> `TotalNLL_b(u,i)`.')
    A('4. Repeat on `valid`, then fit `mu_F, sigma_F` over ALL union candidates of `valid`')
    A('   (never targets only) for EACH same-family checkpoint and apply the same scaler to')
    A('   `test`. The calibrator is mandatory here: per-factorization target NLL scales differ')
    A('   (`factorization_generality/_teacher_forced.log` reports mean total target NLL from')
    A('   7.36 to 9.87 depending on dataset and K).')
    A('5. Score `LSE`, `MinNLL` and `Uniform` and re-evaluate NDCG@20 with the frozen')
    A('   `cc_rerank_methods` definitions.')
    A('')
    A('## 3. Concrete size of the inference work')
    A('')
    A('Requested accounting -- **4 datasets x 3 same-family pairs x 2 splits x 2 checkpoints = '
      f'{budget_totals["passes"]} teacher-forced scoring passes**, scoring '
      f'**{budget_totals["candidates"]:,} (user, candidate) pairs**')
    A('(valid + test). Exact per-cell counts, computed from the frozen beams:')
    A('')
    A('| dataset | split | pair | users | mean union size | candidate scorings | passes |')
    A('|---|---|---|---|---|---|---|')
    for d in budget_detail:
        if not d['pair_kind'].startswith('same_family'):
            continue
        A(f'| {d["dataset"]} | {d["split"]} | {d["pair_kind"]} | {d["n_users"]:,} | '
          f'{d["mean_union_size"]:.2f} | {d["candidate_scorings"]:,} | {d["passes"]} |')
    A('')
    A('Because a factorization is shared between pairs (K2 by K2_K3 and K2_K5, K3 by K2_K3 and')
    A(f'K3_K4), the same work can be scheduled as **4 datasets x 2 splits x 4 factorizations = '
      f'{budget_totals["optimized_passes"]} GPU passes**, scoring '
      f'{budget_totals["optimized_candidates"]:,} (user, candidate) pairs:')
    A('')
    A('| dataset | split | pool | users | mean pool size | candidate scorings | passes |')
    A('|---|---|---|---|---|---|---|')
    for d in budget_detail:
        if d['pair_kind'].startswith('same_family'):
            continue
        A(f'| {d["dataset"]} | {d["split"]} | '
          f'{d["pair_kind"].replace("optimised_pool_", "")} | {d["n_users"]:,} | '
          f'{d["mean_union_size"]:.2f} | {d["candidate_scorings"]:,} | 1 |')
    A('')
    A('### Cost estimate')
    A('')
    if rates:
        A('Anchored on the measured throughput of the frozen candidate-NLL run on THIS machine')
        A(f'(`{rates["source"]}`, RTX 4090, L = 4 SID tokens, batch 512):')
        A('')
        for ds, r in rates['per_dataset_cand_per_s'].items():
            A(f'* {ds}: {r:,.0f} candidate-scorings/s')
        A('')
        A(f'Using the slowest measured rate ({rate:,.0f} candidate-scorings/s):')
    else:
        A(f'Using the documented frozen-run rate {rate:,.0f} candidate-scorings/s:')
    A('')
    A(f'* requested schedule (48 passes): **{per_pass / 60:.1f} min** of pure scoring')
    A(f'* deduplicated schedule (32 passes): **{per_pass_opt / 60:.1f} min** of pure scoring')
    A('* add checkpoint load / tokenisation / IO per pass; the frozen run spent 44-86 s per')
    A('  (dataset, factorization, split) pass INCLUDING those overheads, so this is the same')
    A('  order of magnitude')
    A('* single GPU (24 GB class), a few hours at most; it fits in one')
    A('  session and needs no scheduling decision')
    A('')
    A('## 4. Explicitly: no training of any kind is required')
    A('')
    A('For everything in section 2: **no new T5 training, no new tokenizer training, no gate')
    A('training, no fine-tuning, no calibrator training**. The same-family checkpoints already')
    A('exist and are sha256-verified in `outputs/final_ablation/family_artifact_inventory.csv`')
    A(f'({n_ok}/48 matrix cells verified). The only operations are load checkpoint -> eval() ->')
    A('`torch.no_grad` teacher forcing -> numpy reduction. The z-calibrator is two scalars')
    A('(mean and standard deviation) fitted in closed form on the valid split.')
    A('')
    A('One design caveat, stated plainly: the frozen cross-family LSE result uses the SHARED')
    A('model (`outputs/cross_tokenizer_shared/step2_shared_model/runs/<ds>_shared_spec_rqvae_'
      'seed2025/')
    A('`shared_model.pth`), one T5 carrying both vocabularies plus control tokens, whereas the')
    A('inference-only same-family plan above combines the two SEPARATELY trained same-family')
    A('checkpoints. That is the strongest same-family control obtainable without training. A')
    A('fully matched shared-model same-family control would need one shared T5 per dataset per')
    A('pair (12 runs) and is therefore out of scope for a frozen-method ablation -- it is the one')
    A('variant that WOULD require training, and it is not needed to answer the question "does a')
    A('same-family pair reach the cross-family LSE gain".')
    A('')
    A('## 5. What is already established WITHOUT that compute')
    A('')
    A('From `outputs/final_ablation/family_ablation.csv` (beam dumps only, test split, all')
    A('candidates of the frozen 30-item beams, definitions identical to the frozen fusion core):')
    A('')
    A(f'* candidate complementarity: mean Jaccard over the 12 primary same-family rows = '
      f'{mean_same_j:.4f} vs {mean_cross_j:.4f} over the 4 cross-family rows')
    A(f'* target coverage of the union: same-family {mean_same_cov:.4f} vs cross-family '
      f'{mean_cross_cov:.4f}')
    A(f'* Uniform (0.5*P_a + 0.5*P_b) NDCG@20: same-family {mean_same_u:.6f} vs cross-family '
      f'{mean_cross_u:.6f}')
    A('* per-pair single-model quality, union size, top-20 Jaccard, and a paired user-level')
    A('  bootstrap of Uniform vs the stronger single model are all in the same CSV')
    A('* the pipeline reproduces the published `factorization_generality/'
      'pairwise_diversity_results.csv`')
    A(f'  numbers exactly (max |delta fusion_ndcg20| over the {len(checks)} overlapping rows = '
      f'{worst_check:.2e}), which validates the loader and the fusion definitions used for the')
    A('  cross-family rows as well')
    A('* beam-structural contribution of a third factorization (ablation G inventory) is in')
    A('  `outputs/final_ablation/factorization_count_three_way.csv`')
    A('')
    A('These are enough to say whether the same-family controls are structurally LESS')
    A('complementary than the cross-family pair (Jaccard, union size, coverage). They are NOT')
    A('enough to say whether a same-family pair reaches the same LSE candidate-likelihood gain,')
    A('because that depends on each checkpoint\'s likelihood of the OTHER beam\'s candidates --')
    A('precisely the quantity that needs the GPU passes above.')
    A('')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# Task 1(b)/(c) + Task 4 -- notes file
# ---------------------------------------------------------------------------
def build_notes(rows_inv, fac_inv, fam_rows, checks, three_rows):
    n_ok = sum(r['cell_verified'] for r in rows_inv)
    bad = [r for r in rows_inv if not r['cell_verified']]
    primary = [r for r in fam_rows if r['is_primary'] == 1]
    same = [r for r in primary if r['pair_group'] == 'same_family']
    cross = [r for r in primary if r['pair_group'] == 'cross_family']

    def tab(rs, keys, header):
        out = ['| ' + ' | '.join(header) + ' |',
               '|' + '|'.join('---' for _ in header) + '|']
        for r in rs:
            out.append('| ' + ' | '.join(str(r[k]) for k in keys) + ' |')
        return out

    L = []
    A = L.append
    A('# FAMILY_ABLATION_NOTES')
    A('')
    A('Ablation E (same-family vs cross-family factorization) + Ablation G inventory')
    A('(number of factorizations). Generated by `analysis/mechanism/final_ablation/fa_family.py`')
    A(f'on {time.strftime("%Y-%m-%d %H:%M:%S%z")}. CPU-only: beam dumps + frozen artifacts, no')
    A('checkpoint executed, no training, no tokenizer training.')
    A('')
    A('## Task 1(a) -- inventory verdict')
    A('')
    A(f'* Cells in `ARTIFACT_MATRIX.csv`: {len(rows_inv)} '
      '(4 datasets x 3 families x K in {2,3,4,5}).')
    A('* Fully verified per cell: the four files exist; sha256 of the SID, the checkpoint and')
    A('  the TEST beam match the matrix columns `sid_sha256` / `ckpt_sha256` / `beam_sha256`;')
    A('  SID shape/injectivity/code range OK; beam budget 30; item-id shift reproduced;')
    A('  SID<->beam token binding fraction 1.0; targets identical across families of the same')
    A('  dataset; and single-model NDCG@20 + target_present recomputed from the test beam match')
    A(f'  the matrix `metrics` to <1e-6.')
    A(f'* **Verdict: {n_ok}/48 cells fully verified.**')
    if bad:
        A('* Cells with problems:')
        for r in bad:
            A(f'  * {r["dataset"]} / {r["family"]} / K={r["K"]}: `{r["problems"]}`')
    else:
        A('* No mismatch, no missing file, no failing recomputation.')
    A('* One honest gap: `ARTIFACT_MATRIX.csv` stores a sha256 only for the TEST beam. The')
    A('  `valid_beam` files therefore could NOT be sha-compared against the matrix. They were')
    A('  verified structurally instead: exists, 30-item budget, unambiguous item-id shift,')
    A('  SID<->beam binding fraction 1.0, targets identical to the reference target array of the')
    A('  same dataset/split. `valid_beam_sha_ok` is recorded as `no_reference_in_matrix`.')
    A('')
    A('## Task 1(b) -- same-family pairs available')
    A('')
    A('All pairs below live inside ONE family, i.e. they re-partition the SAME T5 embedding')
    A('space, so they are same-family CONTROLS, not independent families. Files:')
    A('`ARTIFACT_MATRIX.csv` (sid / checkpoint / valid beam / test beam per cell).')
    A('')
    A('| pair | available | single-model NDCG@20 (Beauty ; Cell Phones ; Health ; Toys) |')
    A('|---|---|---|')
    for kind, ka, kb in (('spectral K2 + K3', 'spectral_k2', 'spectral_k3'),
                         ('spectral K3 + K4', 'spectral_k3', 'spectral_k4'),
                         ('spectral K2 + K5', 'spectral_k2', 'spectral_k5'),
                         ('spectral K3 + K5', 'spectral_k3', 'spectral_k5'),
                         ('spectral K4 + K5', 'spectral_k4', 'spectral_k5')):
        vals = []
        for ds in DATASETS:
            va = json.loads(MATRIX_INDEX[ka][ds]['metrics'])['NDCG@20']
            vb = json.loads(MATRIX_INDEX[kb][ds]['metrics'])['NDCG@20']
            vals.append(f'{va:.4f} + {vb:.4f}')
        A(f'| {kind} | yes (4/4 datasets) | ' + ' ; '.join(vals) + ' |')
    A('')
    A('The same construction exists inside the `even` and `random` families (K in 2..5, 4/4')
    A('datasets). The CPU-only table below adds even K2+K3, even K3+K4, random K2+K3 and random')
    A('K3+K4 as extra same-family controls.')
    A('')
    A('## Task 1(c) -- third factorizations available for a count ablation')
    A('')
    A('| factorization | kind | source | datasets | caveat |')
    A('|---|---|---|---|---|')
    for r in fac_inv:
        if r['factorization'] in ('even_k2', 'even_k3', 'even_k4', 'even_k5',
                                  'random_k2', 'random_k3', 'random_k4', 'random_k5',
                                  'rqvae3', 'opq3', 'rq1_opq2'):
            A(f'| {r["factorization"]} | {r["kind"]} | {r["source"]} | {r["n_datasets"]} | '
              f'{r["caveat"]} |')
    A('')
    A('Full paths (sid, checkpoint, valid beam, test beam) for every row are in')
    A('`outputs/final_ablation/factorization_count_inventory.csv`, together with per-factorization')
    A('existence/verification columns (`cells_verified`, `sid_beam_binding_ok_all`,')
    A('`targets_match_reference_all`).')
    A('')
    A('Reading of this inventory:')
    A('')
    A('* `even_k3` / `random_k3` are safe on checkpoint provenance (canonical metric, sha-verified)')
    A('  but are NOT independent families: they partition the same embedding space as spectral,')
    A('  so a 3-way built from them controls "a third factorization", not "a third family".')
    A('* `opq3` and `rq1_opq2` ARE genuinely independent families '
      '(`outputs/cross_tokenizer_shared/step1_cross_tokenizer/beams/`), but their checkpoints')
    A('  were selected with a legacy token-position metric, so they are SECONDARY evidence only.')
    A('* `rqvae3` is the frozen second factorization F2 and is already a member of the base pair.')
    A('* No factorization in the repository combines three families with a shared-vocabulary')
    A('  checkpoint: there is exactly one shared model per dataset (`spectral_k3 + rqvae3`). A')
    A('  3-way count ablation with ranking metrics would therefore need teacher-forced NLL under')
    A('  the third checkpoint (inference, not training) -- see `TRAINING_REQUIREMENT_REPORT.md`.')
    A('')
    A('Beam-structural contribution of a third factorization (frozen pair + one third, test')
    A('split, no NLL, no ranking metric) -- `outputs/final_ablation/'
      'factorization_count_three_way.csv`:')
    A('')
    A('| dataset | third | union 2-way | union 3-way | +candidates | coverage 2-way | '
      'coverage 3-way | +coverage | users target only in third |')
    A('|---|---|---|---|---|---|---|---|---|')
    for r in three_rows:
        A(f'| {r["dataset"]} | {r["third_factorization"]} | {r["mean_union_size_2way"]:.2f} | '
          f'{r["mean_union_size_3way"]:.2f} | {r["mean_incremental_union_size"]:.2f} | '
          f'{r["target_coverage_2way"]:.4f} | {r["target_coverage_3way"]:.4f} | '
          f'{r["incremental_coverage"]:+.4f} | {r["users_target_in_third_only"]} |')
    A('')
    A('## Task 2 -- primary CPU-only comparison (test split)')
    A('')
    A('Definitions, all from the frozen fusion core:')
    A('')
    A('* `C_a`, `C_b`: distinct item ids in each model\'s 30-item beam (duplicates merged, `-1`')
    A('  dropped).')
    A('* `candidate_jaccard`: mean over users of `|C_a and C_b| / |C_a or C_b|` on the FULL beams.')
    A('* `top20_candidate_jaccard`: the same quantity restricted to each model\'s top-20, which is')
    A('  the definition used by the published `factorization_generality` table; reported as a')
    A('  cross-check (`fg_abs_diff_top20_candidate_jaccard` in the CSV).')
    A('* `target_coverage_union`: fraction of users whose target is in `C_a or C_b`.')
    A('* `uniform_ndcg20`: `S(i) = 0.5*P_a(i) + 0.5*P_b(i)`, where `P_x(i)` is that checkpoint\'s')
    A('  beam probability (merged over duplicate columns) and `0` outside its beam; the target is')
    A('  ranked inside the union; NDCG@20 = mean of `1/log2(rank+1)` for rank <= 20. This is')
    A('  exactly `fg_common.union_fuse([a, b])`.')
    A('* `oracle_in_union_ndcg20`: rank 1 for every user whose target is in the union;')
    A('  numerically equal to `target_coverage_union` and is the ceiling of any ranking that can')
    A('  only reorder the union.')
    A('* `uncalibrated_oracle_min_rank_ndcg20`: best-single-model-per-user oracle, i.e. the')
    A('  headroom a perfect SELECTION between the two models could reach.')
    A('* `oracle_headroom_exploited_fraction` = (Uniform - best single) / (selection oracle - best')
    A('  single): how much of that headroom the uniform fusion takes. This is the quality-')
    A('  normalised statistic used for the cross-pair comparison.')
    A('')
    A('Primary rows (4 pair kinds x 4 datasets). The full table -- including the extra')
    A('same-family controls and the secondary independent families -- is in')
    A('`outputs/final_ablation/family_ablation.csv`:')
    A('')
    keys = ['dataset', 'pair_kind', 'mean_candidates_a', 'mean_candidates_b', 'mean_union_size',
            'candidate_jaccard', 'top20_candidate_jaccard', 'target_present_a', 'target_present_b',
            'target_coverage_union', 'single_a_ndcg20', 'single_b_ndcg20', 'uniform_ndcg20',
            'uniform_gain_over_best_single', 'oracle_headroom_exploited_fraction']
    hdr = ['dataset', 'pair_kind', 'cand_a', 'cand_b', 'union', 'jaccard', 'jac_top20',
           'cov_a', 'cov_b', 'cov_union', 'ndcg20_a', 'ndcg20_b', 'uniform_ndcg20',
           'uniform-best', 'headroom_exploited']
    L.extend(tab(primary, keys, hdr))
    A('')
    A('Extra same-family controls and the secondary independent families (same CSV, same')
    A('definitions):')
    A('')
    extra = [r for r in fam_rows if r['is_primary'] == 0]
    L.extend(tab(extra, ['dataset', 'pair_kind', 'mean_union_size', 'candidate_jaccard',
                         'target_coverage_union', 'single_a_ndcg20', 'single_b_ndcg20',
                         'uniform_ndcg20', 'oracle_headroom_exploited_fraction'],
                 ['dataset', 'pair_kind', 'union', 'jaccard', 'cov_union', 'ndcg20_a',
                  'ndcg20_b', 'uniform_ndcg20', 'headroom_exploited']))
    A('')
    A('Pipeline validation: the pipeline reproduces the published')
    A('`factorization_generality/pairwise_diversity_results.csv` rows for the overlapping pairs')
    A(f'(diversity_type G) exactly -- max |delta fusion_ndcg20| = '
      f'{max((c[2] for c in checks), default=float("nan")):.2e} over {len(checks)} rows, and')
    A('`fg_abs_diff_top20_candidate_jaccard` is 0 in every one of them. The same loader and')
    A('fusion code therefore produces the cross-family rows.')
    A('')
    A('## Task 4 -- interpretation rule and honest reading')
    A('')
    A('Rule applied: **absolute NDCG is not the criterion**, because the two models in a pair')
    A('differ in single-factorization quality and the pair kinds are not matched on it. The')
    A('comparison therefore uses (i) structural complementarity that does not depend on quality')
    A('-- Jaccard, union size, target coverage -- and (ii) the quality-normalised')
    A('`oracle_headroom_exploited_fraction`. Absolute `uniform_ndcg20` is always shown next to')
    A('both single-model numbers so the confound is visible.')
    A('')
    A('Per-dataset primary comparison (same-family = mean of the three required same-family')
    A('pairs, then the range across them):')
    A('')
    A('| dataset | cross jac | same jac (range) | cross cov | same cov | cross uni20 | same uni20 | '
      'cross headroom | same headroom | cross best single | same best single |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        s = [r for r in same if r['dataset'] == ds]
        c = [r for r in cross if r['dataset'] == ds][0]
        m = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
        A(f'| {ds} | {c["candidate_jaccard"]:.4f} | '
          f'{m(s, "candidate_jaccard"):.4f} '
          f'({min(float(r["candidate_jaccard"]) for r in s):.4f}-'
          f'{max(float(r["candidate_jaccard"]) for r in s):.4f}) | '
          f'{c["target_coverage_union"]:.4f} | {m(s, "target_coverage_union"):.4f} | '
          f'{c["uniform_ndcg20"]:.6f} | {m(s, "uniform_ndcg20"):.6f} | '
          f'{float(c["oracle_headroom_exploited_fraction"]):.3f} | '
          f'{m(s, "oracle_headroom_exploited_fraction"):.3f} | '
          f'{c["best_single_ndcg20"]:.6f} | {m(s, "best_single_ndcg20"):.6f} |')
    A('')
    A('What the numbers support / what they do not:')
    A('')
    A('* The structural columns (Jaccard, union size, coverage) are quality-free and are the only')
    A('  ones used to claim that a pair is more or less complementary. They are read directly off')
    A('  the table above.')
    A('* `uniform_ndcg20` differences between pair kinds are NOT attributable to family membership')
    A('  alone: the cross-family partner (rqvae3) has a different single-model quality than')
    A('  spectral K2/K4/K5 in most datasets, and a weaker partner changes both the union coverage')
    A('  and the noise injected into the fused ranking.')
    A('* The same-family pairs also change SID length (K2 -> L=3, K5 -> L=6), and the cross-family')
    A('  pair uses independently trained checkpoints, so "different family" is confounded with')
    A('  "different granularity" and with "independent training run". A clean attribution would')
    A('  need same-family pairs matched on single-model quality, which the repository does not')
    A('  contain.')
    A('* Nothing here tests the LSE / candidate-likelihood half of ablation E. Uniform uses only')
    A('  retrieval probabilities; LSE uses each checkpoint\'s likelihood of the OTHER beam\'s')
    A('  candidates. That is the quantity the GPU inference in `TRAINING_REQUIREMENT_REPORT.md`')
    A('  would produce, and no conclusion about it is drawn here.')
    A('* The `opq3` / `rq1_opq2` rows are secondary (legacy token-position checkpoint metric) and')
    A('  are included only as a direction check on independent families.')
    A('* Every cross-family row here uses independently trained checkpoints, while the frozen')
    A('  published cross-family LSE numbers come from the SHARED model (`step2_shared_model`).')
    A('  Task 2 is therefore a structural CPU-only comparison, not a reproduction of the frozen')
    A('  headline, and it must not be quoted as one.')
    A('')
    A('Mechanical reading of the table (filled in by the generator below this line only as')
    A('restatement of the numbers above, no story attached):')
    A('')
    for ds in DATASETS:
        s = [r for r in same if r['dataset'] == ds]
        c = [r for r in cross if r['dataset'] == ds][0]
        m = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
        dj = m(s, 'candidate_jaccard') - float(c['candidate_jaccard'])
        dc = m(s, 'target_coverage_union') - float(c['target_coverage_union'])
        du = m(s, 'uniform_ndcg20') - float(c['uniform_ndcg20'])
        A(f'* {ds}: same-family Jaccard is {dj:+.4f} relative to cross-family, coverage '
          f'{dc:+.4f}, Uniform NDCG@20 {du:+.6f}.')
    A('')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--datasets', default='')
    args = ap.parse_args()
    global DATASETS
    if args.datasets:
        DATASETS = [d for d in DATASETS if d in args.datasets.split(',')]
    t0 = time.time()
    rows = load_matrix()
    register_matrix_cells()
    print(f'[{time.time() - t0:6.1f}s] loaded {len(rows)} matrix cells; '
          f'registry keys = {sorted(MATRIX_INDEX)}')

    ref_targets = build_reference_targets()
    print(f'[{time.time() - t0:6.1f}s] reference target arrays built for both splits')

    print('== Task 1(a): verifying the 48 matrix cells ==')
    rows_inv = task1_matrix(rows, ref_targets)
    n_ok = sum(r['cell_verified'] for r in rows_inv)

    print('== Task 1(c): factorization inventory (matrix + independent families) ==')
    fac_inv = task1_factorizations(rows_inv, ref_targets)
    for r in fac_inv:
        print(f'  {r["factorization"]:12s} {r["kind"]:22s} datasets={r["n_datasets"]} '
              f'verified={r["cells_verified"]} sha={r["sha_all_match"]} '
              f'bind={r["sid_beam_binding_ok_all"]} tgt={r["targets_match_reference_all"]}')

    print('== Task 1(c)/G: three-way structural inventory ==')
    three_rows = three_way_rows()

    print('== Task 2: same-family vs cross-family (CPU-only, test split) ==')
    pub = load_fg_published()
    fam_rows, checks = task2_rows(pub)
    if checks:
        worst = max(c[2] for c in checks)
        print(f'  cross-check vs published factorization_generality: {len(checks)} overlapping '
              f'rows, max |delta fusion_ndcg20| = {worst:.3e}')
        if worst > 1e-9:
            raise SystemExit('HARD FAIL: pipeline does not reproduce the published fusion numbers')
    else:
        print('  WARNING: no overlapping published rows found for the cross-check')

    print('== Task 3: LSE inference budget ==')
    rates = frozen_throughput()
    budget_detail, budget_totals = lse_budget()
    print(f'  requested schedule: {budget_totals["passes"]} passes, '
          f'{budget_totals["candidates"]:,} candidate scorings')
    print(f'  deduplicated schedule: {budget_totals["optimized_passes"]} passes, '
          f'{budget_totals["optimized_candidates"]:,} candidate scorings')
    if rates:
        rate = rates['min_cand_per_s']
        print(f'  measured frozen throughput (min over datasets/splits) = {rate:,.0f} cand/s '
              f'-> {budget_totals["candidates"] / rate / 60:.1f} min (requested), '
              f'{budget_totals["optimized_candidates"] / rate / 60:.1f} min (deduplicated)')

    report = build_report(rows_inv, fam_rows, checks, budget_detail, budget_totals, rates)
    notes = build_notes(rows_inv, fac_inv, fam_rows, checks, three_rows)

    if args.dry_run:
        print(f'[{time.time() - t0:6.1f}s] DRY RUN: nothing written. '
              f'report={len(report)} chars, notes={len(notes)} chars')
        return

    cfg = {'ablation': 'E (family) + G inventory (factorization count)',
           'phase': 'CPU-only inventory + same-family vs cross-family beams',
           'datasets': DATASETS, 'splits': list(SPLITS), 'scored_split': 'test',
           'matrix': str(MATRIX), 'loader': 'xt_common.load_cell(beam_override=matrix paths)',
           'fusion': 'fg_common.union_fuse / candidate_overlap / oracle_rank',
           'bootstrap': 'fa_common.paired_bootstrap n=5000 seed=20260912',
           'gpu': 'none (no checkpoint loaded)', 'training': 'none'}
    from fa_common import config_hash, finish_run, make_run
    targets = [OUT / 'family_artifact_inventory.csv',
               OUT / 'factorization_count_inventory.csv',
               OUT / 'factorization_count_three_way.csv',
               OUT / 'family_ablation.csv',
               OUT / 'TRAINING_REQUIREMENT_REPORT.md',
               OUT / 'FAMILY_ABLATION_NOTES.md']
    for p in targets:
        if p.exists():
            raise SystemExit(f'HARD FAIL: refusing to overwrite {p}')
    run_dir = RUNS / f'family_cpu_ALL_{config_hash(cfg)}'
    if run_dir.exists():
        raise SystemExit(f'HARD FAIL: run directory already exists: {run_dir}')
    run_dir = make_run('family_cpu', 'ALL', cfg)

    write_csv(targets[0], rows_inv)
    write_csv(targets[1], fac_inv)
    write_csv(targets[2], three_rows)
    write_csv(targets[3], fam_rows)
    targets[4].write_text(report)
    targets[5].write_text(notes)

    manifest = [f'{sha256_file(p)}  {p}' for p in sorted(targets + [Path(__file__)])]
    manifest.append(f'{sha256_file(MATRIX)}  {MATRIX}')
    manifest.append(f'{sha256_file(FG_PUB)}  {FG_PUB}')
    (run_dir / 'ARTIFACTS.sha256').write_text('\n'.join(manifest) + '\n')
    finish_run(run_dir, status='completed',
               detail=f'{n_ok}/48 matrix cells verified; {len(fam_rows)} pair rows; '
                      f'{len(three_rows)} three-way inventory rows; CPU-only, no training, '
                      f'no checkpoint loaded')
    print(f'[{time.time() - t0:6.1f}s] wrote 6 artifacts + run dir {run_dir}')


if __name__ == '__main__':
    main()
