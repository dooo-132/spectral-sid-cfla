#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP C/F — Sports candidate union and cross-factorization candidate NLL.

Reuses the audited machinery unchanged:

  * `af_dump_shared.dump_one`      — the frozen shared beam dumper (beams + history repr)
  * `cc_common.candidate_total_nll` — the certified batched teacher-forcing scorer
  * `cc_common.sid_tokens_for_items` — the certified 0-based SID-row tokeniser
  * `cc_candidate_nll.build_union`  — the frozen union rule C_F1 ∪ C_F2, target never added

The only Sports-specific adaptations are:
  * `TOKENIZERS['spectral_k3']['sid']` is shadowed to this round's generated K=3 array,
    because the artifact root holds no K=3 Sports array and is read-only;
  * the union is built from the beam dump directly (the development rounds built theirs
    from a selector feature file that does not exist for Sports).

Subcommands
  beams   dump F1/F2 beams + history repr for a split
  nll     build the union and score every candidate under both factorizations
  scaler  fit the Sports VALID NLL scaler on ALL valid candidates

Usage
  python3 sc_pipeline.py beams  --ckpt <shared_model.pt> --split valid
  python3 sc_pipeline.py nll    --split valid
  python3 sc_pipeline.py scaler
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO),
           str(REPO / 'analysis' / 'mechanism' / 'candidate_conditioned'),
           str(REPO / 'analysis' / 'mechanism' / 'adaptive_factorization'),
           str(REPO / 'analysis' / 'mechanism' / 'factorization_generality'),
           str(REPO / 'analysis' / 'mechanism' / 'cross_tokenizer_shared')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET = 'Sports_and_Outdoors'
F1, F2 = 'spectral_k3', 'rqvae3'
import os
OUT = Path(os.environ.get('SC_OUT', REPO / 'outputs' / 'sports_confirmation'))


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def shadow_sid_for_sports(gen_sid: Path):
    """Point the spectral_k3 arm at this round's generated K=3 array, for Sports only."""
    import xt_common
    tok = xt_common.TOKENIZERS['spectral_k3']
    orig = tok['sid']

    def scoped(ds):
        if ds == DATASET:
            return gen_sid
        return orig(ds)

    tok['sid'] = scoped
    return orig


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------
def cmd_beams(a):
    import torch
    from af_dump_shared import dump_one, load_frozen
    gen_sid = OUT / 'sid' / 'spectral_k3' / 'sid.npy'
    shadow_sid_for_sports(gen_sid)
    from xt_common import shared_vocab_plan
    model, plan = load_frozen(a.ckpt, [F1, F2])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    out = OUT / 'beams'
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in (F1, F2):
        r = dump_one(model, plan, DATASET, f, a.split,
                     out / f'{DATASET}_{f}_{a.split}.npz',
                     out / f'{DATASET}_{f}_{a.split}_hist.npz',
                     a.beam_size, a.infer_size, a.max_len, a.seed, dev)
        res[f] = r
        print(f'  {f}: {json.dumps({k: v for k, v in r.items() if k != "features"})}',
              flush=True)
    (out / f'BEAM_MANIFEST_{a.split}.json').write_text(json.dumps({
        'dataset': DATASET, 'split': a.split, 'checkpoint': a.ckpt,
        'checkpoint_sha256': sha256_file(a.ckpt), 'beam_size': a.beam_size,
        'cells': res}, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# union + NLL
# ---------------------------------------------------------------------------
def cmd_nll(a):
    import pandas as pd
    import torch
    from cc_common import (candidate_total_nll, load_frozen_shared,
                           sid_tokens_for_items)
    from cc_candidate_nll import build_union, ranks_from_probs
    from xt_common import load_sid
    from af_dump_shared import build_cfg  # noqa: F401  (kept for parity of imports)

    split = a.split
    beam_dir = OUT / 'beams'
    cells = {}
    for f in (F1, F2):
        z = np.load(beam_dir / f'{DATASET}_{f}_{split}.npz', allow_pickle=True)
        cells[f] = {'items': np.asarray(z['candidate_item_ids']).astype(np.int64),
                    'probs': np.asarray(z['sequence_scores'], dtype=np.float64)}
    from fg_common import _row_softmax
    for f in (F1, F2):
        c = cells[f]
        c['probs'] = _row_softmax(c['probs'], c['items'] >= 0)

    print(f'[union] building C_F1 ∪ C_F2 for {split}')
    items, up1, up2 = build_union(cells[F1]['items'], cells[F1]['probs'],
                                  cells[F2]['items'], cells[F2]['probs'])
    rank1 = ranks_from_probs(items, up1)
    rank2 = ranks_from_probs(items, up2)
    n, W = items.shape
    owner = np.repeat(np.arange(n), W)
    flat_items = items.reshape(-1)
    keep = flat_items >= 0
    owner = owner[keep]; flat_items = flat_items[keep]
    print(f'  union {len(flat_items)} candidate rows over {n} users '
          f'({len(flat_items)/n:.1f}/user)')

    # frozen shared model + history bank (same code path as the development rounds)
    import cc_common
    cc_common._ART_OVERRIDE = None
    gen_sid = OUT / 'sid' / 'spectral_k3' / 'sid.npy'
    shadow_sid_for_sports(gen_sid)
    model, plan = load_frozen_shared(Path(a.ckpt), device=a.device)

    from cc_common import build_history_bank
    _orig_bank = build_history_bank

    def bank_for(ds, sp, pl):
        # reuse the audited builder, with the spectral SID shadowed for Sports
        import xt_common
        _o = xt_common.TOKENIZERS['spectral_k3']['sid']
        xt_common.TOKENIZERS['spectral_k3']['sid'] = lambda d: (gen_sid if d == DATASET
                                                                else _o(d))
        try:
            return _orig_bank(ds, sp, pl)
        finally:
            xt_common.TOKENIZERS['spectral_k3']['sid'] = _o

    bank = bank_for(DATASET, split, plan)
    nll = {}
    sids = {f: load_sid(f, DATASET) for f in (F1, F2)}
    # load_sid caches by (tok, ds); ensure spectral_k3 for Sports came from our file
    if not np.array_equal(sids[F1], np.load(gen_sid)):
        raise SystemExit('spectral_k3 SID for Sports did not resolve to the generated array')
    for f in (F1, F2):
        toks = sid_tokens_for_items(sids[f], flat_items)
        h, m = bank[f]
        t0 = time.time()
        nll[f] = candidate_total_nll(model, h, m, toks, owner, device=a.device,
                                     batch_size=a.batch_users)
        print(f'  {f:12s} scored {len(flat_items)} candidates '
              f'(finite {np.isfinite(nll[f]).mean():.4f}) in {time.time()-t0:.1f}s',
              flush=True)
        del toks
    del model
    torch.cuda.empty_cache()

    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    users = np.arange(n, dtype=np.int64)
    pd.DataFrame({
        'dataset': DATASET, 'user': users[owner], 'candidate_item': flat_items,
        'source_in_F1_beam': rank1.reshape(-1)[keep] >= 1,
        'source_in_F2_beam': rank2.reshape(-1)[keep] >= 1,
        'rank_F1_if_present': rank1.reshape(-1)[keep],
        'rank_F2_if_present': rank2.reshape(-1)[keep],
        'retrieval_score_F1': up1.reshape(-1)[keep],
        'retrieval_score_F2': up2.reshape(-1)[keep],
    }).to_parquet(d / 'candidate_union.parquet', index=False)
    safe = np.where(flat_items >= 0, flat_items, 0)
    cols = {'dataset': DATASET, 'user': users[owner], 'candidate_item': flat_items,
            'TotalNLL_F1': nll[F1], 'TotalNLL_F2': nll[F2],
            'sid_len_F1': sids[F1].shape[1], 'sid_len_F2': sids[F2].shape[1]}
    for pos in range(sids[F1].shape[1]):
        cols[f'SID_F1_{pos}'] = sids[F1][safe, pos]
    for pos in range(sids[F2].shape[1]):
        cols[f'SID_F2_{pos}'] = sids[F2][safe, pos]
    pd.DataFrame(cols).to_parquet(d / 'candidate_nll.parquet', index=False)
    (d / 'CANDIDATE_PROVENANCE.json').write_text(json.dumps({
        'dataset': DATASET, 'split': split, 'n_users': int(n),
        'n_candidate_rows': int(len(flat_items)),
        'mean_union_size': float(len(flat_items) / n),
        'target_added_to_union': False, 'target_read_by_scorer': False,
        'checkpoint': a.ckpt, 'checkpoint_sha256': sha256_file(a.ckpt),
        'sid_spectral_k3': str(gen_sid), 'sid_spectral_k3_sha256': sha256_file(gen_sid),
        'beam_manifest': str(beam_dir / f'BEAM_MANIFEST_{split}.json'),
        'union_sha256': sha256_file(d / 'candidate_union.parquet'),
        'nll_sha256': sha256_file(d / 'candidate_nll.parquet'),
    }, indent=2, default=str))
    print(f'  wrote {d}/candidate_union.parquet and candidate_nll.parquet')
    return 0


# ---------------------------------------------------------------------------
# scaler
# ---------------------------------------------------------------------------
def cmd_scaler(a):
    import pandas as pd
    nll = pd.read_parquet(OUT / 'valid' / 'candidate_nll.parquet',
                          columns=['TotalNLL_F1', 'TotalNLL_F2'])
    sc = {}
    for key, col in (('F1', 'TotalNLL_F1'), ('F2', 'TotalNLL_F2')):
        v = nll[col].values.astype(np.float64)
        sc[key] = (float(v.mean()), float(v.std(ddof=0)))
    out = {'mu_F1': sc['F1'][0], 'sigma_F1': sc['F1'][1],
           'mu_F2': sc['F2'][0], 'sigma_F2': sc['F2'][1],
           'n_candidates': int(len(nll)),
           'fitted_on': 'ALL Sports validation candidates (not targets only)',
           'protocol': 'harness section 18: Sports VALID fitting is a pre-defined '
                       'unsupervised calibration step, not method retraining'}
    (OUT / 'valid' / 'candidate_nll_scaler.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('beams')
    b.add_argument('--ckpt', required=True)
    b.add_argument('--split', default='valid')
    b.add_argument('--beam_size', type=int, default=30)
    b.add_argument('--infer_size', type=int, default=96)
    b.add_argument('--max_len', type=int, default=20)
    b.add_argument('--seed', type=int, default=2025)
    b.set_defaults(func=cmd_beams)
    n = sub.add_parser('nll')
    n.add_argument('--ckpt', required=True)
    n.add_argument('--split', default='valid')
    n.add_argument('--device', default='cuda')
    n.add_argument('--batch_users', type=int, default=256)
    n.set_defaults(func=cmd_nll)
    s = sub.add_parser('scaler')
    s.set_defaults(func=cmd_scaler)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main())
