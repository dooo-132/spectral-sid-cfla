#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the significance-test results required by the master prompt (section 9)
as plain CSVs at the repository root:

  fusion_vs_single_bootstrap.csv   Uniform multi-granularity vs Best Single K
  dynamic_vs_uniform_bootstrap.csv Dynamic (valid-selected) vs Uniform

Source: the frozen fusion record (produced by the historical fusion driver,
which runs a 5000-resample paired user-level bootstrap over the raw beam dumps).
No recomputation happens here; this is pure export so the numbers cannot drift.
"""
import json
import sys
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils import repo_paths as rp
# evidence JSON lives in the audited repo (read-only)
AUD = rp.ARTIFACT_ROOT / 'frozen_audit'
# new outputs go to this repo
ROOT = rp.OUTPUT_ROOT / 'results' / 'bootstrap'
ROOT.mkdir(parents=True, exist_ok=True)


def main():
    d = json.load(open(AUD / "PHASE5_FUSION.json"))

    vs_single = pd.DataFrame(d["boot_uniform_vs_single"])
    cols = ["dataset", "affinity", "metric", "comparison", "best_single_k",
            "best_single_value", "uniform_value", "delta", "pct_change",
            "ci_lo", "ci_hi", "p_value", "n_users", "n_boot", "significant_95"]
    vs_single = vs_single[cols]
    vs_single.to_csv(ROOT / "fusion_vs_single_bootstrap.csv", index=False)

    dyn = pd.DataFrame(d["boot_dynamic_vs_uniform"])
    cols2 = ["dataset", "affinity", "metric", "comparison", "gamma", "eta",
             "uniform_value", "dynamic_value", "delta", "pct_change",
             "ci_lo", "ci_hi", "p_value", "n_users", "n_boot", "significant_95"]
    dyn = dyn[cols2]
    dyn.to_csv(ROOT / "dynamic_vs_uniform_bootstrap.csv", index=False)

    sel = pd.DataFrame([{k: v for k, v in r.items() if k != "grid"}
                        for r in d["valid_selection"]])
    sel.to_csv(ROOT / "valid_selection_summary.csv", index=False)

    print(f"wrote fusion_vs_single_bootstrap.csv        ({len(vs_single)} rows)")
    print(f"wrote dynamic_vs_uniform_bootstrap.csv      ({len(dyn)} rows)")
    print(f"wrote valid_selection_summary.csv           ({len(sel)} rows)")

    canon = vs_single[(vs_single["affinity"] == "nonneg_pearson_v3")
                      & (vs_single["metric"] == "NDCG@20")]
    print("\n=== PRIMARY CLAIM: uniform multi-granularity vs best single K "
          "(NDCG@20, canonical affinity) ===")
    for _, r in canon.iterrows():
        print(f"  {r['dataset']:32s} best=K{int(r['best_single_k'])} "
              f"{r['best_single_value']:.6f} -> uniform {r['uniform_value']:.6f}  "
              f"delta={r['pct_change']:+.2f}%  "
              f"CI95=[{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]  "
              f"p={r['p_value']:.4g}  significant={r['significant_95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
