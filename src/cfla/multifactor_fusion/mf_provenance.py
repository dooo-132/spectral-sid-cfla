#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_PROVENANCE.json — every script, input, hash and stage of this round in one place.

Usage
  python3 mf_provenance.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import OUT, read_json, sha256_file, write_json  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

STAGES = [
    ('state_recovery', 'mf_state_recovery.py', ['frozen_state.md',
                                                '_frozen_state_INPUTS.json']),
    ('artifact_inventory', 'mf_artifact_matrix.py', ['FACTORISATION_ARTIFACT_MATRIX.csv',
                                                     '_ARTIFACT_MATRIX_DETAIL.json']),
    ('dynamic_fusion_reproduction', 'mf_dynamic_fusion.py',
     ['DYNAMIC_FUSION_REPRODUCTION.json']),
    ('union_build', 'mf_build_union.py', ['validation/valid_union_stats.csv',
                                          'test/test_union_stats.csv']),
    ('oom_preflight', 'mf_preflight.py', ['OOM_PREFLIGHT.md', 'OOM_PREFLIGHT.json',
                                          'gpu_memory_log.csv']),
    ('reuse_audit', 'mf_reuse_audit.py', ['REUSE_AUDIT.json']),
    ('candidate_nll', 'mf_candidate_nll.py',
     ['validation/_VALID_SPECTRAL4_NLL_PROVENANCE.json',
      'validation/_VALID_MULTI5_NLL_PROVENANCE.json',
      'test/_TEST_SPECTRAL4_NLL_PROVENANCE.json',
      'test/_TEST_MULTI5_NLL_PROVENANCE.json']),
    ('baselines', 'mf_stage.py --stage baselines',
     ['validation/validation_nll_scalers.json',
      'validation/validation_baseline_selection.json',
      'validation/validation_results.csv']),
    ('gate_training', 'mf_stage.py --stage gate',
     ['validation/gate_training.json', 'runs/gate_spectral4/gate.pt',
      'runs/gate_multi5/gate.pt']),
    ('freeze', 'mf_stage.py --stage freeze', ['MULTIFACTOR_FUSION_FREEZE_MANIFEST.json']),
    ('test_evaluation', 'mf_stage.py --stage test',
     ['test/results.csv', 'test/bootstrap.csv']),
    ('coverage', 'mf_stage.py --stage coverage',
     ['test/coverage.csv', 'test/rqvae_added_value.csv',
      'test/unique_factorisation_groups.csv']),
    ('efficiency', 'mf_stage.py --stage efficiency', ['test/efficiency.csv']),
    ('final_report', 'mf_final_report.py',
     ['FINAL_MULTIFACTOR_FUSION_REPORT.md', '_REPORT_VERDICTS.json']),
    ('immutability_pre', 'mf_immutability.py --snapshot pre',
     ['PRE_EXISTING_CODE_MANIFEST.sha256', 'PRE_RUN_GIT_STATUS.txt']),
    ('immutability_post', 'mf_immutability.py --snapshot post',
     ['POST_EXISTING_CODE_MANIFEST.sha256', 'POST_RUN_GIT_STATUS.txt',
      'IMMUTABILITY_VERIFICATION.json']),
]

SOURCES = [
    'mf_common.py', 'mf_models.py', 'mf_core.py', 'mf_gate.py', 'mf_eval.py',
    'mf_stage.py', 'mf_state_recovery.py', 'mf_artifact_matrix.py',
    'mf_build_union.py', 'mf_candidate_nll.py', 'mf_dynamic_fusion.py',
    'mf_preflight.py', 'mf_reuse_audit.py', 'mf_immutability.py',
    'mf_final_report.py', 'mf_provenance.py', 'tests/run_tests.py',
]


def git(*a):
    return subprocess.run(['git', *a], cwd=str(REPO), capture_output=True,
                          text=True).stdout.strip()


def main():
    prov = {
        'round': 'Multi-Factorization Candidate-Level Fusion',
        'branch': git('branch', '--show-current'),
        'head': git('rev-parse', 'HEAD'),
        'working_tree': git('status', '--short'),
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'output_root': str(OUT),
        'code_root': str(HERE),
        'append_only': True,
        # `existing_code_unchanged` is True when NO program file was changed by this
        # round, so the "modified" flag is its negation.
        'existing_program_files_modified_by_this_round': (
            not read_json(OUT / 'IMMUTABILITY_VERIFICATION.json')['existing_code_unchanged'])
        if (OUT / 'IMMUTABILITY_VERIFICATION.json').exists() else None,
        'immutability_evidence': {
            'manifest_pre': str(OUT / 'PRE_EXISTING_CODE_MANIFEST.sha256'),
            'manifest_post': str(OUT / 'POST_EXISTING_CODE_MANIFEST.sha256'),
            'verification': str(OUT / 'IMMUTABILITY_VERIFICATION.json'),
            'other_session_commits_touching_snapshotted_files': sorted({
                v.get('commit') for v in
                (read_json(OUT / 'IMMUTABILITY_VERIFICATION.json').get('attribution') or {}).values()
                if v.get('commit')}) if (OUT / 'IMMUTABILITY_VERIFICATION.json').exists() else []},
        'sports_untouched': True,
        'base_models_trained_this_round': 0,
        'sid_tokenizers_trained_this_round': 0,
        'gates_trained_this_round': 2,
        'sources': {s: {'path': str(HERE / s), 'sha256': sha256_file(HERE / s)}
                    for s in SOURCES},
        'stages': [],
    }
    for stage, cmd, outs in STAGES:
        entry = {'stage': stage, 'command': f'python3 experiments/multifactor_fusion/{cmd}',
                 'outputs': {}}
        for o in outs:
            p = OUT / o
            entry['outputs'][o] = {'exists': p.exists(),
                                   'sha256': sha256_file(p) if p.exists() else None,
                                   'bytes': p.stat().st_size if p.exists() else None}
        prov['stages'].append(entry)

    # external inputs, hashed
    ext = {}
    for name, p in (
            ('candidate_conditioned_nll_run',
             REPO / 'outputs/candidate_conditioned/runs/multi_candidate_nll_seed2025_20260914T194657'),
            ('router_user_split',
             REPO / 'outputs/disagreement_gate/splits/router_user_split.csv'),
            ('shared_model_beams',
             REPO / 'outputs/adaptive_factorization/selector/features/beams')):
        if p.is_dir():
            files = sorted(x for x in p.rglob('*') if x.is_file())
            ext[name] = {'dir': str(p), 'n_files': len(files),
                         'sha256_of_listing': sha256_file(p / files[0]) if files else None}
        else:
            ext[name] = {'path': str(p), 'sha256': sha256_file(p)}
    prov['external_inputs'] = ext
    write_json(OUT / '_PROVENANCE.json', prov)
    missing = [f'{s["stage"]}:{k}' for s in prov['stages'] for k, v in s['outputs'].items()
               if not v['exists']]
    print(f'wrote {OUT / "_PROVENANCE.json"}')
    if missing:
        print('MISSING OUTPUTS:')
        for m in missing:
            print('  ', m)
    return 0


if __name__ == '__main__':
    sys.exit(main())
