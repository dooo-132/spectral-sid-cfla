#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 — Beauty, single-seed multi-granularity replication + seed-dependence check.

Answers the reviewer question in the strongest form that the available compute
supports:

  (1) Does the paper's central claim (uniform multi-granularity fusion beats the
      best single granularity) replicate on a SECOND, fully independent seed?
      Compare seed 2026 (K=2,3,4,5) against its own best single K.
  (2) Is the multi-granularity ensemble seed-dependent?  Compare the seed-2025
      and seed-2026 multi-granularity ensembles directly.
  (3) Same-granularity reference at K=2: a multi-seed ensemble at a FIXED
      granularity, evaluated where seeds exist.

All ensembles are equal-weight item-space averaging of per-model beam-softmax
distributions, i.e. exactly the paper's fusion rule; only the composition of the
four models differs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion.common import _row_softmax, canonicalize_item_ids, per_user_ndcg, summarize  # noqa: E402

OUT = Path(__file__).resolve().parent
CANON_DIR = ROOT / "model" / "beam_dumps_Beauty_nonneg_pearson_v3"
SEED_DIR = ROOT / "experiments" / "seed_ensemble" / "beams"
DS = "Beauty"
K_LIST = [2, 3, 4, 5]
N_BOOT = 5000
RNG = 20260912
METRICS = [("Recall", 20), ("NDCG", 20), ("NDCG", 10)]


def canon_path(k, split="test"):
    return CANON_DIR / f"{split}_k{k}_sid{k+1}.npz"


def seed_path(seed, k, split="test"):
    return SEED_DIR / f"{DS}_seed{seed}_{split}_k{k}_sid{k+1}.npz"


def load(p):
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def build_union(dumps):
    n, B = dumps[0]["candidate_item_ids"].shape
    target = canonicalize_item_ids(dumps[0]["target_item_ids"])
    cand = [canonicalize_item_ids(d["candidate_item_ids"]) for d in dumps]
    probs = [_row_softmax(np.asarray(d["sequence_scores"], dtype=np.float64),
                          np.asarray(d["candidate_item_ids"]) >= 0) for d in dumps]
    mm = len(dumps)
    umax = mm * B
    items = np.full((n, umax), -1, dtype=np.int64)
    st = np.zeros((n, umax, mm))
    for i in range(n):
        pos, p = {}, 0
        for m in range(mm):
            for j in range(B):
                it = int(cand[m][i, j])
                if it < 0:
                    continue
                if it not in pos:
                    pos[it] = p
                    items[i, p] = it
                    p += 1
                st[i, pos[it], m] += probs[m][i, j]
    tp = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        w = np.where(items[i] == target[i])[0]
        if len(w):
            tp[i] = w[0]
    return items, st, target, tp, items >= 0


def ev(dumps):
    U = build_union(dumps)
    n, _, mm = U[1].shape
    pu = per_user_ndcg(U[0], U[1], U[3], U[4], np.ones((n, mm)) / mm)
    return pu, summarize(pu), float(U[4].sum(axis=1).mean())


def boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return dict(delta=0.0, ci_lo=0.0, ci_hi=0.0, p_value=float("nan"), note="identical")
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_value=float(min(1.0, max(p, 1 / N_BOOT))),
                pct_change=float(100 * d.mean() / b.mean()) if b.mean() else float("nan"))


def main():
    report = {"dataset": DS, "n_boot": N_BOOT, "results": {}, "bootstrap": []}

    # ---------------- (1) seed 2026 multi-granularity ----------------
    dumps26 = [load(seed_path(2026, k)) for k in K_LIST]
    if any(d is None for d in dumps26):
        miss = [str(seed_path(2026, k).relative_to(ROOT)) for k, d in zip(K_LIST, dumps26) if d is None]
        print("MISSING seed-2026 beams:")
        for m in miss:
            print("   ", m)
        return 2
    pu26, s26, u26 = ev(dumps26)

    # best single K for seed 2026
    single26 = {}
    for k in K_LIST:
        pu, sm, _ = ev([load(seed_path(2026, k))])
        single26[k] = {"summary": sm, "pu": pu}
    bk26 = max(K_LIST, key=lambda k: single26[k]["summary"]["NDCG@20"])

    # ---------------- (2) seed 2025 (canonical) ----------------
    dumps25 = [load(canon_path(k)) for k in K_LIST]
    pu25, s25, u25 = ev(dumps25)
    single25 = {}
    for k in K_LIST:
        pu, sm, _ = ev([load(canon_path(k))])
        single25[k] = {"summary": sm, "pu": pu}
    bk25 = max(K_LIST, key=lambda k: single25[k]["summary"]["NDCG@20"])

    report["results"] = {
        "seed2026_multi_ensemble": s26, "seed2026_best_single_k": bk26,
        "seed2026_single": {str(k): v["summary"] for k, v in single26.items()},
        "seed2026_union_size": u26,
        "seed2025_multi_ensemble": s25, "seed2025_best_single_k": bk25,
        "seed2025_single": {str(k): v["summary"] for k, v in single25.items()},
        "seed2025_union_size": u25,
    }

    # bootstraps
    def add(name, pa, pb, meta=None):
        for metric, kk in METRICS:
            key = f"{metric}@{kk}"
            r = boot(pa[key], pb[key])
            row = {"comparison": name, "metric": key,
                   "a_value": float(pa[key].mean()), "b_value": float(pb[key].mean()), **r,
                   "n_boot": N_BOOT,
                   "significant_95": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)}
            if meta:
                row.update(meta)
            report["bootstrap"].append(row)

    add("seed2026_multi_vs_best_single", pu26, single26[bk26]["pu"], {"best_single_k": bk26})
    add("seed2025_multi_vs_best_single", pu25, single25[bk25]["pu"], {"best_single_k": bk25})
    add("multi_ensemble_seed2026_vs_seed2025", pu26, pu25)

    # ---------------- (3) same-granularity at K=2 if extra seeds exist --------
    k2_seeds = [2025] + [s for s in (2026, 2027, 2028) if load(seed_path(s, 2)) is not None]
    if len(k2_seeds) >= 2:
        dumps_same = [load(canon_path(2)) if s == 2025 else load(seed_path(s, 2))
                      for s in k2_seeds]
        pu_same, s_same, u_same = ev(dumps_same)
        report["results"]["same_granularity_K2_ensemble"] = s_same
        report["results"]["same_granularity_K2_seeds"] = k2_seeds
        report["results"]["same_granularity_K2_union_size"] = u_same
        # compare against the multi-granularity ensemble of the SAME seed set is
        # not possible (K!=2 models for those seeds), so compare against the
        # seed-2025 multi-granularity 4-model ensemble instead and report sizes.
        add("same_granularity_K2_vs_seed2025_multi", pu_same, pu25,
            {"n_models_same": len(k2_seeds), "n_models_multi": 4})

    (OUT / "PHASE6B_M1_BEAUTY.json").write_text(json.dumps(report, indent=2, default=str))
    rows = []
    for b in report["bootstrap"]:
        rows.append(b)
    pd.DataFrame(rows).to_csv(ROOT / "same_vs_multi_granularity_bootstrap.csv", index=False)

    print("=" * 78)
    print("BEAUTY — multi-granularity replication across seeds")
    print("=" * 78)
    print(f"  seed 2025 multi-granularity NDCG@20 = {s25['NDCG@20']:.6f} "
          f"(best single K={bk25}: {single25[bk25]['summary']['NDCG@20']:.6f})"
          f"  gain={100*(s25['NDCG@20']/single25[bk25]['summary']['NDCG@20']-1):+.2f}%")
    print(f"  seed 2026 multi-granularity NDCG@20 = {s26['NDCG@20']:.6f} "
          f"(best single K={bk26}: {single26[bk26]['summary']['NDCG@20']:.6f})"
          f"  gain={100*(s26['NDCG@20']/single26[bk26]['summary']['NDCG@20']-1):+.2f}%")
    print(f"  union sizes: seed2025={u25:.1f}  seed2026={u26:.1f}")
    print("\n  bootstraps:")
    for b in report["bootstrap"]:
        if b["metric"] != "NDCG@20":
            continue
        pc = b.get("pct_change")
        print(f"    {b['comparison']:44s} d={b['delta']:+.6f} "
              f"({pc:+.2f}% ) p={b['p_value']:.4g} sig={b['significant_95']}"
              if pc is not None and pc == pc else f"    {b['comparison']:44s} identical")
    print(f"\n  wrote {OUT/'PHASE6B_M1_BEAUTY.json'}")
    print(f"  wrote same_vs_multi_granularity_bootstrap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
