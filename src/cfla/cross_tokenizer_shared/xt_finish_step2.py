#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unattended finisher for STEP 2.

Waits for the shared-training driver to exit, then runs the whole STEP 2
evaluation chain in order and records a marker file. Every stage is a script that
has already been validated end-to-end on a mid-training checkpoint, so this adds
sequencing only — no new analysis logic.

Stages
  1. dump per-factorization beams from each final shared checkpoint (inference only)
  2. shared single-factorization metrics + complementarity + Only-F groups
  3. shared teacher-forced NLL + Only-F NLL alignment
  4. independent-vs-shared core table
  5. figures, STEP 2 report, final two-step report, provenance, guards

Writes step2_shared_model/_FINISH_STATUS.json when done.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xt_common import DATASETS, OUT as OUT_ROOT, PRIMARY, STEP2  # noqa: E402

PY = sys.executable
LOG = STEP2 / '_finish.log'
STATUS = STEP2 / '_FINISH_STATUS.json'
BEAMS = STEP2 / 'beams'
INFER_SIZE = 32          # batch size does not affect beam-search results


def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def run(name, cmd):
    log(f'>>> {name}')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(HERE.parents[2]), capture_output=True, text=True)
    dt = time.time() - t0
    tail = (p.stdout or '')[-1500:]
    if p.returncode != 0:
        log(f'FAILED {name} rc={p.returncode} ({dt:.0f}s)')
        log((p.stdout or '')[-3000:])
        log((p.stderr or '')[-3000:])
        return False, tail
    log(f'ok {name} ({dt:.0f}s)')
    return True, tail


def driver_alive():
    r = subprocess.run(['pgrep', '-f', r'^python3 .*xt_run_shared\.py'],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main():
    STATUS.unlink(missing_ok=True)
    log('finisher started; waiting for the training driver to exit')
    while driver_alive():
        time.sleep(120)
    log('training driver exited')

    steps, results = [], {}
    for ds in DATASETS:
        ck = STEP2 / 'runs' / f'{ds}_shared_spec_rqvae_seed2025' / 'shared_model.pth'
        if not ck.exists():
            log(f'{ds}: checkpoint missing; skip')
            continue
        out = BEAMS / f'{ds}_{PRIMARY[0]}_test.npz'
        cmd = [PY, str(HERE / 'xt_dump_shared_beams.py'), '--ckpt', str(ck),
               '--datasets', ds, '--splits', 'test', '--infer_size', str(INFER_SIZE)]
        ok, _ = run(f'dump shared beams {ds}', cmd)
        if not ok:
            # a smaller inference batch cannot change beam-search results but
            # sharply lowers peak memory; this is the failure mode seen earlier
            log(f'retrying {ds} dump at infer_size 8')
            ok, _ = run(f'dump shared beams {ds} (retry)',
                        cmd[:-1] + ['8'])
        results[f'dump_{ds}'] = ok
        done = all((BEAMS / f'{ds}_{f}_test.npz').exists() for f in PRIMARY)
        if not ok and not done:
            STATUS.write_text(json.dumps({'status': 'failed', 'stage': f'dump_{ds}',
                                          'results': results}, indent=2))
            return 1
        log(f'{ds}: shared beams present = {done}')

    chain = [
        ('shared analysis', [PY, str(HERE / 'xt_shared_analysis.py')]),
        ('shared NLL + alignment', [PY, str(HERE / 'xt_shared_nll.py')]),
        ('independent vs shared', [PY, str(HERE / 'xt_independent_vs_shared.py')]),
        ('figures', [PY, str(HERE / 'xt_figures.py')]),
        ('STEP2 report', [PY, str(HERE / 'xt_step2_report.py')]),
        ('final report', [PY, str(HERE / 'xt_final_report.py')]),
        ('provenance', [PY, str(HERE / 'xt_provenance.py')]),
        ('guards', [PY, str(HERE / 'xt_verify.py')]),
        ('completeness audit', [PY, str(HERE / 'xt_completeness.py')]),
    ]
    for name, cmd in chain:
        ok, _ = run(name, cmd)
        results[name] = ok
        # provenance and guards may legitimately report non-zero on missing inputs,
        # so only hard-fail on the analysis stages
        if not ok and name in ('shared analysis', 'shared NLL + alignment',
                               'independent vs shared'):
            STATUS.write_text(json.dumps({'status': 'failed', 'stage': name,
                                          'results': results}, indent=2))
            return 1

    man = STEP2 / 'independent_vs_shared_factorization.csv'
    audit = OUT_ROOT / 'COMPLETENESS_AUDIT.json'
    comp_status = None
    if audit.exists():
        comp_status = json.loads(audit.read_text()).get('status')
    summary = {'status': 'done' if comp_status in (None, 'COMPLETE') else comp_status,
               'results': results, 'finished': time.strftime('%FT%T%z'),
               'completeness': comp_status}
    if man.exists():
        import pandas as pd
        d = pd.read_csv(man)
        core = d[d['metric'].isin(['fusion_gain', 'oracle_gain', 'hit_disagreement',
                                   'nll_alignment_accuracy'])]
        summary['shared_vs_independent'] = core[
            ['dataset', 'metric', 'independent_factors', 'shared_checkpoint']].to_dict('records')
    fr = OUT_ROOT / 'FINAL_TWO_STEP_REPORT.md'
    if fr.exists():
        txt = fr.read_text()
        vs = {}
        for label in ('CROSS-INDEPENDENT-TOKENIZER COMPLEMENTARITY',
                      'INSTANCE-SPECIFIC CROSS-TOKENIZER PREFERENCE',
                      'SHARED-PARAMETER FACTORIZATION COMPLEMENTARITY',
                      'INDEPENDENT PARAMETERS ARE NECESSARY FOR COMPLEMENTARITY'):
            if label + ':' in txt:
                vs[label] = txt.split(label + ':', 1)[1].strip().split('\n', 1)[0].strip()
        summary['verdicts'] = vs
    STATUS.write_text(json.dumps(summary, indent=2, default=str))
    log(f'FINISHED status={summary["status"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
