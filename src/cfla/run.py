#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the CFLA pipeline end to end on user-produced artifacts.

    python3 -m src.cfla.run --config configs/cfla_pool.example.yaml

Inputs (all produced by the shipped stages; see docs/PIPELINE_STAGES.md):
  * one SID array per factorization  (``sid``, (n_items, L) int64, row i = item i+1)
  * one T5 checkpoint per factorization (``ckpt``)
  * one beam dump per factorization and split (``beam_valid`` / ``beam_test``)
  * the dataset parquet splits (``dataset_dir/{valid,test}.parquet``)

Outputs (under ``--output-dir`` or the config's ``output_dir``):
  * metrics.json      - per-method Recall@K / NDCG@K, calibration statistics,
                        candidate-pool statistics and paired bootstrap results
  * per_user.csv      - per-user rank / hit information for every method
  * run_config.json   - the resolved config actually used

The ground-truth target is used ONLY for metric computation after scoring; it is
never inserted into the candidate union and never used by the scorer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.cfla import cfla as C  # noqa: E402

METHODS = ("cfla", "uniform", "minnll")


def check_paths(cfg):
    """Return a list of missing input paths (files only)."""
    missing = []
    ds = Path(cfg["dataset_dir"])
    for split in ("valid", "test"):
        p = ds / f"{split}.parquet"
        if not p.is_file():
            missing.append(str(p))
    for f in cfg["factorizations"]:
        for key in ("sid", "ckpt", "beam_valid", "beam_test"):
            if not Path(f[key]).is_file():
                missing.append(f"{f['name']}.{key}: {f[key]}")
    return missing


def score_split(cfg, split, device, batch_size):
    """Union + per-factorization candidate NLL for one split."""
    tqdm = _tqdm()
    fact = cfg["factorizations"]
    dumps, n_users = C.load_dumps(fact, split)
    union = C.build_union(dumps)
    user_idx, item_ids = C.flat_pairs(union)
    parquet = str(Path(cfg["dataset_dir"]) / f"{split}.parquet")
    nll_by_factor, sid_lens = {}, {}
    for f in tqdm(fact, desc=f"[{split}] factorizations"):
        item_to_tokens = C.load_item_to_tokens(f["sid"], cfg["codebook_size"])
        sid_lens[f["name"]] = len(next(iter(item_to_tokens.values())))
        histories = C.load_eval_histories(parquet, item_to_tokens,
                                          int(cfg["max_len"]), n_users)
        model, vocab = C.load_model(f["ckpt"], device=device)
        expected = sid_lens[f["name"]] * int(cfg["codebook_size"]) + 1
        if int(vocab) != expected:
            raise SystemExit(
                f"{f['name']}: checkpoint vocab_size={vocab} but SID length "
                f"{sid_lens[f['name']]} x codebook {cfg['codebook_size']} + 1 = "
                f"{expected}; the SID array and checkpoint do not match")
        nll_by_factor[f["name"]] = C.candidate_nll(
            model, histories, user_idx, item_ids, item_to_tokens,
            device=device, batch_size=batch_size)
        del model
    prob_matrix = np.zeros((len(item_ids), len(fact)), dtype=np.float64)
    offset = 0
    for i, p in enumerate(union["probs"]):
        prob_matrix[offset:offset + len(p)] = p
        offset += len(p)
    return {"union": union, "n_users": n_users,
            "nll": nll_by_factor, "probs": prob_matrix,
            "sid_lens": sid_lens, "n_pairs": len(item_ids)}


def _tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **kw):
            return x
        return tqdm


def build_scores(fact_names, block, scalers, methods):
    """Score every candidate row for the requested methods."""
    z = np.stack([C.zscore(block["nll"][n], scalers[n]) for n in fact_names], axis=1)
    out = {}
    if "cfla" in methods:
        out["cfla"] = C.score_cfla(z)
    if "minnll" in methods:
        out["minnll"] = C.score_minnll(z)
    if "uniform" in methods:
        out["uniform"] = C.score_uniform(block["probs"])
    return out, z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run CFLA (candidate union -> teacher-forced NLL -> "
                    "validation calibration -> log-sum-exp aggregation -> "
                    "evaluation) on user-produced artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="pool config YAML (see configs/cfla_pool.example.yaml)")
    ap.add_argument("--splits", default="valid,test",
                    help="splits used for calibration/evaluation (default valid,test)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="rows per teacher-forcing batch (default 256)")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help=f"comma-separated subset of {METHODS}")
    ap.add_argument("--bootstrap", type=int, default=5000,
                    help="paired user-level bootstrap resamples (0 disables)")
    ap.add_argument("--output-dir", default=None,
                    help="output directory (default: config output_dir or <repo>/outputs/cfla)")
    ap.add_argument("--check", action="store_true",
                    help="only validate that every input path exists")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty output directory")
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown methods: {bad} (choose from {METHODS})")

    cfg = C.load_config(args.config)
    missing = check_paths(cfg)
    if args.check:
        print(f"config: {args.config}")
        print(f"factorizations: {[f['name'] for f in cfg['factorizations']]}")
        print(f"checked {len(cfg['factorizations']) * 4 + 2} input paths, "
              f"{len(missing)} missing")
        for m in missing:
            print("MISSING", m)
        return 1 if missing else 0
    if missing:
        raise SystemExit("missing input artifacts:\n  " + "\n  ".join(missing) +
                         "\nRun the earlier pipeline stages first "
                         "(docs/PIPELINE_STAGES.md) or fix the config paths.")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "valid" not in splits:
        raise SystemExit("the `valid` split is required to fit the calibration "
                         "statistics (they must never be fitted on test)")
    if "test" not in splits:
        print("WARNING: no `test` split requested; only calibration statistics "
              "will be reported")

    out_dir = Path(args.output_dir or cfg.get("output_dir") or (REPO / "outputs" / "cfla"))
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {out_dir}; pass "
                         f"--overwrite or choose another --output-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = {}
    for split in splits:
        print(f"=== [{split}] building candidate union and candidate NLL ===")
        blocks[split] = score_split(cfg, split, args.device, args.batch_size)
        b = blocks[split]
        print(f"  users={b['n_users']}  candidate rows={b['n_pairs']}  "
              f"sid lengths={b['sid_lens']}")

    fact_names = [f["name"] for f in cfg["factorizations"]]
    scalers = C.fit_scalers(blocks["valid"]["nll"])
    print("=== calibration (fitted on ALL validation candidates) ===")
    for n in fact_names:
        mu, sd = scalers[n]
        print(f"  {n}: mu={mu:.6f} sigma={sd:.6f}")

    report = {"config": str(args.config), "dataset_dir": cfg["dataset_dir"],
              "factorizations": fact_names, "sid_lengths": blocks["valid"]["sid_lens"],
              "calibration": {n: {"mu": scalers[n][0], "sigma": scalers[n][1]}
                              for n in fact_names},
              "splits": {}, "methods": methods}
    per_user_rows = []
    for split in splits:
        block = blocks[split]
        scores, z = build_scores(fact_names, block, scalers, methods)
        report["splits"][split] = {
            "n_users": block["n_users"], "n_candidate_rows": block["n_pairs"],
            "union_coverage": float(block["union"]["target_pos"].mean() >= 0),
            "metrics": {},
        }
        for m in methods:
            um = C.per_user_metrics(block["union"], scores[m])
            report["splits"][split]["metrics"][m] = C.summarize(um)
            for i in range(block["n_users"]):
                per_user_rows.append({"split": split, "method": m, "user": i,
                                      "rank": int(um["rank"][i]),
                                      "in_union": bool(um["in_union"][i])})
        # paired bootstrap of CFLA vs uniform (if both available)
        if "cfla" in methods and "uniform" in methods and args.bootstrap:
            um_c = C.per_user_metrics(block["union"], scores["cfla"])
            um_u = C.per_user_metrics(block["union"], scores["uniform"])
            report["splits"][split]["cfla_vs_uniform"] = {
                metric: C.paired_bootstrap(um_c[metric], um_u[metric],
                                           n_boot=args.bootstrap)
                for metric in ("Recall@10", "NDCG@10", "NDCG@20")}

    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    with open(out_dir / "per_user.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_user_rows[0]))
        w.writeheader()
        w.writerows(per_user_rows)

    print("=== results ===")
    for split in splits:
        print(f"[{split}]")
        for m in methods:
            metrics = report["splits"][split]["metrics"][m]
            print(f"  {m:8s} " + "  ".join(
                f"{k}={v:.5f}" for k, v in metrics.items()
                if k in ("Recall@5", "Recall@10", "NDCG@10", "NDCG@20")))
    print(f"wrote {out_dir/'metrics.json'}, {out_dir/'per_user.csv'}, "
          f"{out_dir/'run_config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
