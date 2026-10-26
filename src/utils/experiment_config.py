#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified experiment config schema, loading, validation and run identity.

Config layout
-------------
    experiment:
      name:      <str>          human label, used in the run id
      dataset:   <str>          must exist under ARTIFACT_ROOT/data/
      seed:      <int>
    paths:
      artifact_root: <str>      optional; read-only evidence repo
      output_root:   <str>      optional; where this run writes
    sid:
      method:         spectral
      affinity:       nonneg_pearson | abs_pearson | cosine
      k:              int >= 2
      codebook_size:  256
      alignment_mode: canonical | item_id
    model:
      batch_size, learning_rate, max_epochs, patience, max_len, infer_size
    generation:
      beam_size
    evaluation:
      mode: item | token
    fusion:
      method: uniform

Canonical values are NOT re-typed here: `load_config` can fill model/training
defaults from `configs/canonical.yaml`, which was migrated verbatim from the
audited `CANONICAL_CONFIG.yaml`.

Validation runs BEFORE any work: schema, paths, SID length, vocab, codebook,
dataset and alignment are all checked, and a bad config exits immediately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from . import repo_paths as rp

SID_METHODS = ('spectral',)
EVAL_MODES = ('item', 'token')
FUSION_METHODS = ('uniform', 'grid_search', 'structure_aware', 'confidence_only')


class ConfigError(Exception):
    """Raised on any invalid or inconsistent configuration."""


def canonical_defaults() -> dict:
    """Model/training/generation defaults read from configs/canonical.yaml."""
    p = rp.REPO_ROOT / 'configs' / 'canonical.yaml'
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text()) or {}
    model = d.get('model', {})
    train = d.get('training', {})
    infer = d.get('inference', {})
    return {
        'model': {
            'batch_size': train.get('batch_size', 256),
            'learning_rate': train.get('lr', 1e-4),
            'max_epochs': train.get('num_epochs', 200),
            'patience': train.get('early_stop_patience', 10),
            'max_len': train.get('max_len', 20),
            'infer_size': train.get('infer_size', 96),
        },
        'generation': {'beam_size': infer.get('beam_size', 30)},
        'evaluation': {'mode': 'item'},
        'fusion': {'method': 'uniform'},
        'sid': {'method': 'spectral', 'codebook_size': 256,
                'affinity': 'nonneg_pearson', 'alignment_mode': 'canonical'},
    }


def load_config(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f'config not found: {path}')
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f'config must be a mapping, got {type(raw).__name__}')

    cfg = canonical_defaults()
    # deep-merge the user's config over the canonical defaults
    for section, val in raw.items():
        if isinstance(val, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(val)
        else:
            cfg[section] = val
    cfg['_source_path'] = str(path)
    cfg['_source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def validate(cfg: dict) -> dict:
    """Validate and normalise. Raises ConfigError on the first problem."""
    errs = []

    exp = cfg.get('experiment') or {}
    dataset = exp.get('dataset')
    if not dataset:
        errs.append('experiment.dataset is required')
    elif not (rp.ARTIFACT_ROOT / 'data' / dataset).is_dir():
        errs.append(f"experiment.dataset {dataset!r} not found under "
                    f"{rp.ARTIFACT_ROOT / 'data'}")
    if not isinstance(exp.get('seed', 2025), int):
        errs.append('experiment.seed must be an integer')

    sid = cfg.get('sid') or {}
    if sid.get('method', 'spectral') not in SID_METHODS:
        errs.append(f"sid.method must be one of {SID_METHODS}")
    if sid.get('affinity', 'nonneg_pearson') not in (
            'nonneg_pearson', 'abs_pearson', 'cosine',
            'nonneg_pearson_v3', 'abspearson_v3'):
        errs.append('sid.affinity is not a known affinity')
    k = sid.get('k')
    if not isinstance(k, int) or k < 2:
        errs.append(f'sid.k must be an integer >= 2, got {k!r}')
    cb = sid.get('codebook_size', 256)
    if not isinstance(cb, int) or cb <= 0:
        errs.append(f'sid.codebook_size must be a positive integer, got {cb!r}')
    am = sid.get('alignment_mode', 'canonical')
    if am not in ('canonical', 'item_id'):
        errs.append(f"sid.alignment_mode must be 'canonical' or 'item_id', got {am!r}")

    # SID length / vocab consistency
    if isinstance(k, int) and k >= 2:
        L = k + 1
        if cfg['sid'].get('sid_length') not in (None, L):
            errs.append(f'sid.sid_length must equal K+1 = {L}')
        if isinstance(cb, int) and cb > 0:
            vocab = cb * L + 1
            if cfg['sid'].get('vocab_size') not in (None, vocab):
                errs.append(f'sid.vocab_size must equal codebook*L+1 = {vocab}')
        cfg['sid']['sid_length'] = L
        if isinstance(cb, int) and cb > 0:
            cfg['sid']['vocab_size'] = cb * L + 1

    ev = (cfg.get('evaluation') or {}).get('mode', 'item')
    if ev not in EVAL_MODES:
        errs.append(f"evaluation.mode must be one of {EVAL_MODES}, got {ev!r}")
    fu = (cfg.get('fusion') or {}).get('method', 'uniform')
    if fu not in FUSION_METHODS:
        errs.append(f"fusion.method must be one of {FUSION_METHODS}, got {fu!r}")
    bs = (cfg.get('generation') or {}).get('beam_size', 30)
    if not isinstance(bs, int) or bs <= 0:
        errs.append(f'generation.beam_size must be a positive integer, got {bs!r}')

    # alignment feasibility: item_id needs an id column
    if am == 'item_id' and dataset and (rp.ARTIFACT_ROOT / 'data' / dataset).is_dir():
        import pandas as pd
        try:
            emb = pd.read_parquet(rp.item_embedding(dataset))
            from src.sid.embedding import alignment_report
            rep = alignment_report(emb)
            if not rep.get('available'):
                errs.append(
                    f"alignment_mode='item_id' requires an item-id column in "
                    f"{rp.item_embedding(dataset).name}; none found")
            elif not rep.get('is_bijection_1_to_N'):
                errs.append(
                    f"alignment_mode='item_id' requires item ids 1..N; got "
                    f"min={rep['id_min']} max={rep['id_max']} "
                    f"unique={rep['n_unique_ids']} n={rep['n_items']}")
        except Exception as e:                      # pragma: no cover
            errs.append(f'could not inspect the embedding file: {e}')

    # path sanity
    for section, key in (('paths', 'artifact_root'), ('paths', 'output_root')):
        v = (cfg.get(section) or {}).get(key)
        if v and not Path(v).is_absolute():
            errs.append(f'{section}.{key} must be an absolute path, got {v!r}')

    if errs:
        raise ConfigError('invalid config:\n  - ' + '\n  - '.join(errs))
    return cfg


def run_id(cfg: dict) -> str:
    """<dataset>_<method>_<affinity>_k<K>_seed<S>_<short-config-hash>"""
    exp = cfg['experiment']
    sid = cfg['sid']
    ds = str(exp['dataset']).replace(' ', '_')
    payload = json.dumps({
        'dataset': exp['dataset'], 'seed': exp.get('seed', 2025),
        'sid': {kk: sid.get(kk) for kk in
                ('method', 'affinity', 'k', 'codebook_size', 'alignment_mode', 'pca_dim')},
        'model': cfg.get('model', {}),
        'generation': cfg.get('generation', {}),
        'evaluation': cfg.get('evaluation', {}),
        'fusion': cfg.get('fusion', {}),
    }, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return (f"{ds}_{sid.get('method', 'spectral')}_{sid.get('affinity')}"
            f"_k{sid.get('k')}_seed{exp.get('seed', 2025)}_{h}")


def run_dir(cfg: dict) -> Path:
    return rp.OUTPUT_ROOT / 'runs' / run_id(cfg)
