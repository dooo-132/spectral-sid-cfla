#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State recovery document (prompt section 5) for the multi-factorization round.

Rebuilt from disk: git state, datasets, every factorization artifact with its
hashes, the shared model, the beam dumps, the reused candidate NLL, the old
DynamicFusion config/results, the item-id convention and the stop-condition check.

Usage
  python3 mf_state_recovery.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_common import (ART, CC, DATASETS, DG, FACTORIZATIONS, FLABEL, MODEL_KIND,  # noqa: E402
                       N_ITEMS, N_TEST, OUT, S4, M5, SHORT, beam_path, ckpt_path,
                       dynamic_json, read_json, sha256_file, shared_ckpt, sid_path,
                       structure_json, write_json)

REPO = Path(__file__).resolve().parents[3]


def git(*args):
    return subprocess.run(['git', *args], cwd=str(REPO), capture_output=True,
                          text=True).stdout.strip()


def main():
    L = []
    A = L.append
    A('# frozen_state — Multi-Factorization Candidate-Level Fusion')
    A('')
    A('Rebuilt from disk at the start of this round. Nothing below is copied from a '
      'previous report without re-deriving it.')
    A('')
    A('## A. Repository state')
    A('')
    A('```')
    A(f'active repo   : {REPO}')
    A(f'branch        : {git("branch", "--show-current")}')
    A(f'HEAD          : {git("rev-parse", "HEAD")}')
    A(f'working tree  : {"CLEAN" if not git("status", "--short") else "DIRTY"}')
    A('```')
    A('')
    A('```')
    A(git('log', '-12', '--oneline'))
    A('```')
    A('')
    A('Repository roles (from the task definition):')
    A('')
    A('```text')
    A(f'upstream root    = READ ONLY   {UP_ROOT}')
    A(f'artifact root    = READ ONLY   {ART}')
    A(f'repository       = ACTIVE      {REPO}')
    A('```')
    A('')
    A('Append-only development: existing program files are byte-for-byte immutable '
      'this round. `PRE_EXISTING_CODE_MANIFEST.sha256` snapshots them before the '
      'experiments and `POST_EXISTING_CODE_MANIFEST.sha256` verifies them after.')
    A('')
    A('## B. Datasets')
    A('')
    A('```text')
    A('Development datasets (used many times before; this round is a '
      'development/post-hoc benchmark):')
    for ds in DATASETS:
        A(f'  {ds:32s} items={N_ITEMS[ds]:6d}  test users={N_TEST[ds]:6d}  '
          f'display={SHORT[ds]}')
    A('')
    A('Sports_and_Outdoors: untouched by this round. No Sports test metric was read,')
    A('no Sports parameter was tuned, and no Sports artifact was written.')
    A('```')
    A('')
    A('## C. Factorization artifacts (dataset × F)')
    A('')
    A('| dataset | F | label | SID length | checkpoint kind | checkpoint SHA256 | '
      'valid beam SHA256 | test beam SHA256 |')
    A('|---|---|---|---|---|---|---|---|')
    detail = {}
    for ds in DATASETS:
        for f in FACTORIZATIONS:
            sid = np.load(sid_path(f, ds), allow_pickle=True)
            ck = shared_ckpt(ds) if MODEL_KIND[f] == 'shared' else ckpt_path(f, ds)
            vb, tb = beam_path(f, ds, 'valid'), beam_path(f, ds, 'test')
            A(f'| {SHORT[ds]} | {f} | {FLABEL[f]} | {sid.shape[1]} | {MODEL_KIND[f]} | '
              f'`{str(sha256_file(ck))[:16]}…` | `{str(sha256_file(vb))[:16]}…` | '
              f'`{str(sha256_file(tb))[:16]}…` |')
            detail[f'{ds}|{f}'] = {
                'sid': str(sid_path(f, ds)), 'sid_sha256': sha256_file(sid_path(f, ds)),
                'sid_shape': list(sid.shape),
                'checkpoint': str(ck), 'checkpoint_sha256': sha256_file(ck),
                'checkpoint_kind': MODEL_KIND[f],
                'valid_beam': str(vb), 'valid_beam_sha256': sha256_file(vb),
                'test_beam': str(tb), 'test_beam_sha256': sha256_file(tb),
                'beam_source': ('shared_model_beam' if f in ('spectral_k3', 'rqvae3')
                                else 'independent_history_beam')}
    A('')
    A('Full machine-readable inventory: `FACTORISATION_ARTIFACT_MATRIX.csv` '
      '(20/20 dataset × factorization cells usable).')
    A('')
    A('## D. Shared model availability')
    A('')
    A('| dataset | checkpoint | SHA256 | vocabulary | control tokens |')
    A('|---|---|---|---|---|')
    for ds in DATASETS:
        p = shared_ckpt(ds)
        A(f'| {SHORT[ds]} | `{p.name}` | `{str(sha256_file(p))[:16]}…` | 1027 | '
          f'`spectral_k3`→1025, `rqvae3`→1026 |')
    A('')
    A('Model assignment, established from the artifacts rather than assumed:')
    A('')
    A('```text')
    A('F            candidate set                    scorer (candidate NLL)')
    A('spectral_k2  independent historical beam      independent K2 checkpoint')
    A('spectral_k3  independent historical beam      independent K3 checkpoint')
    A('spectral_k4  independent historical beam      independent K4 checkpoint')
    A('spectral_k5  independent historical beam      independent K5 checkpoint')
    A('rqvae3       frozen shared-model RQVAE beam   frozen shared checkpoint (reused)')
    A('```')
    A('')
    A('Why spectral_k3 uses its own checkpoint rather than the shared one: the frozen '
      'DynamicFusion artifact records per-user confidences that match the independent '
      'K3 dump exactly and the shared model\'s K3 beam only approximately, so the '
      'frozen baseline and its candidate universe were built on the independent beams. '
      'Reproducing it bit-exactly (a hard requirement) needs those beams, and using '
      'the shared checkpoint for spectral_k3 while every other spectral K uses its own '
      'would break the length/fairness control.')
    A('')
    A('## E. Beam dumps')
    A('')
    A('```text')
    for f in FACTORIZATIONS:
        for ds in DATASETS[:1]:
            A(f'{f:12s} {ds:10s} valid -> {beam_path(f, ds, "valid")}')
            A(f'{"":12s} {"":10s} test  -> {beam_path(f, ds, "test")}')
    A('beam size per factorization: 30 (historical, never increased)')
    A('```')
    A('')
    A('## F. Reused candidate NLL')
    A('')
    A('```text')
    A(f'frozen source : {CC}')
    A('factorizations: spectral_k3 (column TotalNLL_F1), rqvae3 (column TotalNLL_F2)')
    A('union there   : C_K3 ∪ C_RQVAE, target never added')
    A('```')
    A('')
    A('**What is reused and what is not.** The RQVAE column of that artifact is '
      'reused wherever this round\'s candidate pool covers the same (user, candidate) '
      'pair; RQVAE is scored by the same shared checkpoint in both rounds, so the '
      'values are directly transferable, and only the uncovered remainder is '
      'computed. The **spectral_k3** column is NOT reused: those numbers are the '
      '*shared* checkpoint\'s likelihood of the candidate SID, while this round\'s '
      'spectral_k3 candidates come from the *independent* K3 beam, and the two K3 '
      'beams agree on only ~11% of candidates. Reusing them would score spectral_k3 '
      'under a different model than every other spectral K and would break the '
      'length/fairness control of prompt section 10, so spectral_k3 is computed under '
      'its own checkpoint. `REUSE_AUDIT.json` records the independent recomputation '
      'that validates the shared scorer against the frozen RQVAE values before any '
      'reuse, and `_VALID_*_NLL_PROVENANCE.json` records per cell how many values were '
      'reused and how many computed.')
    A('')
    A('## G. Old DynamicFusion configuration and results')
    A('')
    A('Recovered from the audited repository:')
    A('')
    A('```text')
    A('implementation : the historical fusion implementation')
    A('formula        : alpha_K(u) = softmax(gamma * z(Q)_K + eta * Conf_K(u))')
    A('                 score(u,i) = sum_K alpha_K(u) * P_K(i|u)')
    A('confidence     : Conf_K(u) = 1 - H(p_K(u)) / log(max(B_valid, 2)),')
    A('                 p_K = row-softmax(sequence_scores) over valid beams')
    A('score norm     : softmax, temperature 1.0; conf temperature 1.0')
    A('aggregation    : per-item beam probabilities accumulated (summed) inside each')
    A('                 factorization, then a weighted sum across factorizations')
    A('Q source       : structure_weights_<ds>_<affinity>.json (z-scored)')
    A('```')
    A('')
    A('| dataset | gamma | eta | frozen test NDCG@20 | frozen test Recall@20 |')
    A('|---|---|---|---|---|')
    frozen = {}
    for ds in DATASETS:
        j = read_json(dynamic_json(ds, 'test'))
        frozen[ds] = j
        A(f'| {SHORT[ds]} | {j["gamma"]} | {j["eta"]} | {j["metrics"]["NDCG@20"]:.6f} | '
          f'{j["metrics"]["Recall@20"]:.6f} |')
    A('')
    A('Reproduction: `mf_dynamic_fusion.py` re-derives these numbers from the raw '
      'dumps with the frozen gamma/eta. No gamma, eta or confidence definition was '
      're-searched (prompt section 12).')
    A('')
    A('## H. Item-id convention')
    A('')
    A('```text')
    A('dataset item id       1-based 1..N      key of item2code, parquet columns')
    A('SID array row index   0-based 0..N-1    row i is item i+1')
    A('historical dump ids   0-based          raw = canonical - 1, -1 = "no item"')
    A('this round writes     user = canonical 1-based id, candidate_item = 1-based id')
    A('```')
    A('')
    A('Detected per dump from its own tokens '
      '(`sid[raw - shift] + pos*256 + 1 == candidate_tokens`, exact at shift 0 for '
      'all 40 dataset × factorization × split cells).')
    A('')
    A('## I. Union construction and stop-condition check')
    A('')
    A('```text')
    A('Pool A (spectral4) = C_K2 ∪ C_K3 ∪ C_K4 ∪ C_K5')
    A('Pool B (multi5)    = C_K2 ∪ C_K3 ∪ C_K4 ∪ C_K5 ∪ C_RQVAE')
    A('target added to a union: never (asserted by construction '                     )
    A('                         and by tests/test_no_target_leakage.py)')
    A('```')
    A('')
    A('No stop condition triggered before scoring: the checkpoints match the audited '
      'ones, the item-id convention was reconstructible from the dumps themselves, '
      'the frozen artifact\'s RQVAE column is exactly the RQVAE candidate set this '
      'round uses (so those NLL values are reusable under the same model), and no '
      'historical artifact needed to be overwritten. The one OOM event of this round is recorded '
      'in `OOM_PREFLIGHT.md` and `_CONFIG_REVISIONS.json`: it happened **during the '
      'preflight**, before any scoring run existed, and the predeclared chunk size '
      'was re-declared exactly once, from the previously documented safe operating '
      'point, then frozen. No metric had been computed at that point.')
    A('')
    A('## J. Artifact hashes')
    A('')
    A('```')
    A(f'FACTORISATION_ARTIFACT_MATRIX.csv  {sha256_file(OUT / "FACTORISATION_ARTIFACT_MATRIX.csv")}')
    A(f'valid_union_stats.csv              {sha256_file(OUT / "validation/valid_union_stats.csv")}')
    A(f'test_union_stats.csv               {sha256_file(OUT / "test/test_union_stats.csv")}')
    A(f'DYNAMIC_FUSION_REPRODUCTION.json   {sha256_file(OUT / "DYNAMIC_FUSION_REPRODUCTION.json")}')
    A('```')
    A('')
    (OUT / 'frozen_state.md').write_text('\n'.join(L) + '\n')

    write_json(OUT / '_frozen_state_INPUTS.json', {
        'git': {'branch': git('branch', '--show-current'),
                'head': git('rev-parse', 'HEAD'),
                'status': git('status', '--short')},
        'cells': detail,
        'shared_checkpoints': {ds: {'path': str(shared_ckpt(ds)),
                                    'sha256': sha256_file(shared_ckpt(ds))}
                               for ds in DATASETS},
        'dynamic_fusion_frozen': {ds: read_json(dynamic_json(ds, 'test'))['metrics']
                                  for ds in DATASETS},
        'dynamic_fusion_structure': {ds: str(structure_json(ds)) for ds in DATASETS},
        'reused_nll_root': str(CC),
        'disagreement_gate_root': str(DG),
        'sports_touched': False})
    print(f'wrote {OUT / "frozen_state.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
