#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble _PROVENANCE.json for the factorization-generality study.

Records, for every input artifact, its path and sha256; for every output, its
sha256; the git state; the comparability audit; and the explicit statement of
what was and was not run (no training).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fg_common import OUT, REPO, SEED_ROOT, DATASETS, FAMILIES  # noqa: E402
from src.utils import repo_paths as rp  # noqa: E402


def sha256(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=str(REPO), capture_output=True,
                          text=True).stdout.strip()


def main():
    import pandas as pd
    art = pd.read_csv(OUT / 'ARTIFACT_MATRIX.csv')
    seed = pd.read_csv(OUT / 'SEED_ARTIFACT_MATRIX.csv')
    audit = json.loads((OUT / '_AUDIT.json').read_text())
    tf = json.loads((OUT / '_TF_PROVENANCE.json').read_text())

    inputs = {'canonical_cells': {}, 'seed_checkpoints': {}, 'teacher_forcing': {}}
    for _, r in art.iterrows():
        inputs['canonical_cells'][f"{r['dataset']}|{r['family']}|K{r['K']}"] = {
            'sid': {'path': r['sid'], 'sha256': sha256(r['sid'])},
            'checkpoint': {'path': r['checkpoint'], 'sha256': sha256(r['checkpoint'])},
            'test_beam': {'path': r['test_beam'], 'sha256': sha256(r['test_beam'])},
            'valid_beam': {'path': r['valid_beam'], 'sha256': sha256(r['valid_beam'])},
            'raw_item_id_shift': int(r['raw_item_id_shift']) if 'raw_item_id_shift' in r else None,
            'usable': int(r['usable']),
        }
    def _s(v):
        return '' if v is None or (isinstance(v, float)) else str(v)

    for _, r in seed.iterrows():
        tb, vb = _s(r['test_beam']), _s(r['valid_beam'])
        inputs['seed_checkpoints'][
            f"{r['dataset']}|K{r['K']}|seed{r['seed']}"] = {
            'checkpoint': {'path': str(r['checkpoint']),
                           'sha256': sha256(r['checkpoint'])},
            'test_beam': {'path': tb, 'sha256': sha256(tb) if tb else None},
            'valid_beam': {'path': vb, 'sha256': sha256(vb) if vb else None},
        }
    inputs['teacher_forcing'] = tf.get('inputs', {})

    outputs = {}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ('_PROVENANCE.json',):
            outputs[str(p.relative_to(OUT))] = {
                'bytes': p.stat().st_size, 'sha256': sha256(p)}

    prov = {
        'study': 'factorization_generality',
        'question': ('is multi-granularity complementarity granularity-specific, '
                     'or the generic effect of any alternative discrete '
                     'factorization of item identity?'),
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'repository': str(REPO),
        'artifact_root': str(rp.ARTIFACT_ROOT),
        'artifact_root_readonly': True,
        'git': {'branch': git('branch', '--show-current'),
                'sha': git('rev-parse', 'HEAD'),
                'base_branch': 'research/full-id-factorization',
                'base_sha': 'unknown',
                'dirty_at_provenance_write': bool(git('status', '--porcelain'))},
        'canonical_config': {
            'datasets': DATASETS,
            'K': [2, 3, 4, 5],
            'families': FAMILIES,
            'alignment_mode': 'canonical',
            'affinity': 'nonneg_pearson_v3 (spectral)',
            'evaluation': 'item-level, beam size 30',
            'fusion': 'uniform item-level probability averaging, no tuning',
            'canonical_seed': 2025,
        },
        'governance': {
            'training_performed': 0,
            'new_models_trained': 0,
            'checkpoints_modified': 0,
            'audited_repo_written': False,
            'canonical_sid_modified': False,
            'old_beam_dumps_overwritten': False,
            'hyperparameters_tuned': 0,
            'test_set_selection': False,
            'paper_read_or_written': False,
            'teacher_forced_nll_used_as_inference_signal': False,
            'beam_dumps_generated_from_existing_seed_checkpoints': 8,
            'beam_dump_generation_is_inference_only': True,
        },
        'comparability_audit': audit,
        'inputs': inputs,
        'outputs': outputs,
    }
    (OUT / '_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))
    print(f'wrote _PROVENANCE.json ({len(outputs)} outputs, '
          f'{len(inputs["canonical_cells"])} canonical cells, '
          f'{len(inputs["seed_checkpoints"])} seed checkpoints)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
