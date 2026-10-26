#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Item-level evaluation of one beam dump.

Uses exactly the canonical functions in `src.fusion.common` (the same functions
used by `src/runner.py`): canonical 0-based -> 1-based item-id canonicalisation,
length-normalised sequence scores -> per-row retrieval probabilities, beam
merging per item, item-level Recall@/NDCG@ for K in {5,10,20}.

The metric definition is item-level (`eval_mode=item`), the canonical one for
this project.  Nothing is trained and no beam is generated here.

Usage
-----
    python3 scripts/evaluate.py --dump outputs/beam_dumps/Beauty/test_k2_sid3.npz
    python3 scripts/evaluate.py --dump ... --out outputs/metrics/test_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fusion.common import (  # noqa: E402
    _row_softmax, canonicalize_item_ids, per_user_ndcg, summarize,
)


def evaluate_dump(dump: np.lib.npyio.NpzFile) -> dict:
    items = canonicalize_item_ids(dump['candidate_item_ids'])
    tgt = canonicalize_item_ids(dump['target_item_ids'])
    scores = np.asarray(dump['sequence_scores'], dtype=np.float64)
    valid = items >= 0
    probs = _row_softmax(scores, valid)
    n, beam = items.shape

    # merge beams that decode to the same item id (sum of probabilities)
    union = np.full((n, beam), -1, np.int64)
    tensor = np.zeros((n, beam, 1))
    for i in range(n):
        pos = {}
        p = 0
        for j in range(beam):
            it = int(items[i, j])
            if it < 0:
                continue
            if it not in pos:
                pos[it] = p
                union[i, p] = it
                p += 1
            tensor[i, pos[it], 0] += probs[i, j]
    target_pos = np.full(n, -1, np.int64)
    for i in range(n):
        w = np.where(union[i] == tgt[i])[0]
        if len(w):
            target_pos[i] = int(w[0])
    per_user = per_user_ndcg(union, tensor, target_pos, union >= 0,
                             np.ones((n, 1)))
    metrics = summarize(per_user)
    metrics['n_users'] = int(n)
    metrics['beam_size'] = int(beam)
    metrics['target_in_union_rate'] = float((target_pos >= 0).mean())
    if 'split' in dump.files:
        metrics['split'] = str(dump['split'])
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True, help='beam dump .npz to evaluate')
    ap.add_argument('--out', default=None,
                    help='optional metrics JSON path; default: print only')
    a = ap.parse_args(argv)
    dump_path = Path(a.dump)
    if not dump_path.is_file():
        print(f'FATAL: beam dump not found: {dump_path}', file=sys.stderr)
        return 2
    metrics = evaluate_dump(np.load(dump_path, allow_pickle=True))
    text = json.dumps(metrics, indent=2)
    print(text)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n')
        print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
