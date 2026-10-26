#!/usr/bin/env python3
"""Generate RQ-OPQ SIDs for TIGER with an optional collision-resolution token.

Supported methods
-----------------
1. rq1_opq2: one RQ-KMeans token, followed by two OPQ/PQ tokens on the RQ residual.
2. opq3:      three OPQ/PQ tokens directly on the original item embeddings.

The primary output .npy contains three quantization codes plus one collision-resolution
code, all in [0, codebook_size-1]. A separate ``*_raw3.npy`` file preserves the
original three-token SID. Do NOT add layer offsets here: the isolated TIGER adapter
adds ``layer * codebook_size + 1`` at load time.

This implementation follows the public OneSearch code structure:
- RQ: FAISS KMeans followed by residual subtraction.
- OPQ: ``OPQ{M},IVF1,PQ{M}x{bits}`` using FAISS index_factory.

Unlike the original sample, inverted-list vector IDs are explicitly read and
sorted before codes are returned, guaranteeing alignment with item rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as exc:  # pragma: no cover - executed in target environment
    raise SystemExit(
        "FAISS is required. Install one build only, for example: pip install faiss-cpu"
    ) from exc


LOGGER = logging.getLogger("rq_opq_sid")


@dataclass
class SIDMetrics:
    num_items: int
    sid_length: int
    codebook_size: int
    unique_full_sids: int
    collision_items: int
    collision_rate: float
    singleton_items: int
    icr: float
    per_layer_unique: List[int]
    per_layer_cur: List[float]
    min_code: int
    max_code: int


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _to_float32_matrix(values: Iterable[object]) -> np.ndarray:
    rows: List[np.ndarray] = []
    expected_dim: int | None = None
    for row_idx, value in enumerate(values):
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
        elif hasattr(value, "as_py"):
            arr = np.asarray(value.as_py())
        else:
            raise TypeError(
                f"Unsupported embedding type at row {row_idx}: {type(value)!r}"
            )
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if expected_dim is None:
            expected_dim = int(arr.shape[0])
        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at row {row_idx}: "
                f"got {arr.shape[0]}, expected {expected_dim}"
            )
        rows.append(arr)
    if not rows:
        raise ValueError("No embeddings found")
    matrix = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0].tolist()
        raise ValueError(f"Embedding matrix contains NaN/Inf, first bad position={bad}")
    return matrix


def infer_columns(df: pd.DataFrame, item_id_col: str | None, embedding_col: str | None) -> Tuple[str, str]:
    if item_id_col is None:
        candidates = ["ItemID", "item_id", "itemID", "id"]
        item_id_col = next((c for c in candidates if c in df.columns), None)
    if embedding_col is None:
        candidates = ["embedding", "emb", "item_embedding", "vector"]
        embedding_col = next((c for c in candidates if c in df.columns), None)
    if item_id_col is None or item_id_col not in df.columns:
        raise ValueError(
            f"Cannot infer item id column. Available columns: {list(df.columns)}. "
            "Pass --item_id_col explicitly."
        )
    if embedding_col is None or embedding_col not in df.columns:
        raise ValueError(
            f"Cannot infer embedding column. Available columns: {list(df.columns)}. "
            "Pass --embedding_col explicitly."
        )
    return item_id_col, embedding_col


def load_item_embeddings(
    parquet_path: Path,
    item_id_col: str | None,
    embedding_col: str | None,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {parquet_path}")
    LOGGER.info("Loading item embeddings: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    item_id_col, embedding_col = infer_columns(df, item_id_col, embedding_col)

    if df[item_id_col].isna().any():
        raise ValueError(f"{item_id_col} contains null values")
    item_ids = pd.to_numeric(df[item_id_col], errors="raise").astype(np.int64).to_numpy()
    if len(np.unique(item_ids)) != len(item_ids):
        duplicates = pd.Series(item_ids)[pd.Series(item_ids).duplicated()].head().tolist()
        raise ValueError(f"Duplicate item IDs found, examples: {duplicates}")

    order = np.argsort(item_ids, kind="stable")
    item_ids = item_ids[order]
    sorted_embeddings = df.iloc[order][embedding_col]
    embeddings = _to_float32_matrix(sorted_embeddings)

    expected_ids = np.arange(1, len(item_ids) + 1, dtype=np.int64)
    if not np.array_equal(item_ids, expected_ids):
        mismatch = int(np.flatnonzero(item_ids != expected_ids)[0])
        raise ValueError(
            "TIGER dataset.py maps row index+1 to item ID, so ItemID must be contiguous "
            f"1..N after sorting. First mismatch: position={mismatch}, "
            f"expected={expected_ids[mismatch]}, got={item_ids[mismatch]}."
        )

    LOGGER.info("Loaded N=%d, dim=%d", embeddings.shape[0], embeddings.shape[1])
    return item_ids, embeddings, item_id_col, embedding_col


def maybe_l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, 1e-12), dtype=np.float32)


def train_rq_kmeans(
    x: np.ndarray,
    codebook_size: int,
    niter: int,
    seed: int,
    normalize_residual: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    n, d = x.shape
    if n < codebook_size:
        raise ValueError(
            f"RQ KMeans needs at least codebook_size points: N={n}, K={codebook_size}"
        )
    LOGGER.info("Training RQ KMeans: N=%d d=%d K=%d niter=%d", n, d, codebook_size, niter)
    kmeans = faiss.Kmeans(
        d,
        codebook_size,
        niter=niter,
        verbose=True,
        seed=seed,
        max_points_per_centroid=10000,
        min_points_per_centroid=1,
    )
    kmeans.train(x)
    _, assignments = kmeans.index.search(x, 1)
    rq_codes = assignments.reshape(-1).astype(np.int64)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(codebook_size, d)
    residual = np.ascontiguousarray(x - centroids[rq_codes], dtype=np.float32)
    if normalize_residual:
        residual = maybe_l2_normalize(residual)
    LOGGER.info(
        "RQ complete: used=%d/%d, residual mean L2=%.6f",
        len(np.unique(rq_codes)),
        codebook_size,
        float(np.linalg.norm(residual, axis=1).mean()),
    )
    return rq_codes, centroids, residual, kmeans.index


def train_rq_kmeans_2level(
    x: np.ndarray,
    codebook_size: int,
    niter: int,
    seed: int,
    normalize_residual: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[object, object], np.ndarray]:
    """PATCH(rq2_opq3): TWO residual-quantisation levels instead of one.

    Level 1 is exactly ``train_rq_kmeans``. Level 2 repeats the same call on the
    level-1 residual, with the SAME hyper-parameters (niter, seed, codebook_size
    and the same faiss.Kmeans settings). The returned ``residual`` is the
    level-2 residual, which is what the OPQ stage then quantises.

    This is the ordinary multi-level residual-quantisation construction (the same
    one ``rqvae/main.py`` uses with ``num_emb_list=[256]*n``); no other part of the
    recipe is touched.
    """
    c1, cent1, res1, idx1 = train_rq_kmeans(
        x=x, codebook_size=codebook_size, niter=niter, seed=seed,
        normalize_residual=normalize_residual)
    c2, cent2, res2, idx2 = train_rq_kmeans(
        x=res1, codebook_size=codebook_size, niter=niter, seed=seed,
        normalize_residual=normalize_residual)
    codes = np.column_stack([c1, c2]).astype(np.int64)
    return codes, cent2, res2, (idx1, idx2), cent1


def _decode_packed_codes(
    packed: np.ndarray,
    n_codebook: int,
    bits_per_codebook: int,
) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    result = np.empty((packed.shape[0], n_codebook), dtype=np.int64)
    n_bytes = packed.shape[1]
    for i in range(packed.shape[0]):
        reader = faiss.BitstringReader(faiss.swig_ptr(packed[i]), n_bytes)
        for j in range(n_codebook):
            result[i, j] = int(reader.read(bits_per_codebook))
    return result


def train_and_encode_opq_source_style(
    x: np.ndarray,
    n_codebook: int,
    codebook_size: int,
) -> Tuple[np.ndarray, object, str]:
    n, d = x.shape
    bits_float = math.log2(codebook_size)
    bits = int(bits_float)
    if 2**bits != codebook_size:
        raise ValueError("codebook_size must be a power of two for PQ bit packing")
    if d % n_codebook != 0:
        raise ValueError(
            f"Embedding dim {d} must be divisible by OPQ codebook count {n_codebook}"
        )
    if n < codebook_size:
        raise ValueError(
            f"OPQ/PQ needs at least codebook_size points: N={n}, K={codebook_size}"
        )

    factory = f"OPQ{n_codebook},IVF1,PQ{n_codebook}x{bits}"
    LOGGER.info("Training source-style OPQ index: %s, N=%d d=%d", factory, n, d)
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
    index.train(x)
    index.reset()
    index.add(x)

    # The public sample reads codes from IVF list 0. We additionally retrieve stored
    # vector IDs and restore original row order, preventing silent row/code mismatch.
    ivf_index = faiss.downcast_index(index.index)
    extracted_ivf = faiss.extract_index_ivf(ivf_index)
    invlists = extracted_ivf.invlists
    list_size = int(invlists.list_size(0))
    if list_size != n:
        raise RuntimeError(f"IVF1 contains {list_size} vectors, expected {n}")

    packed = faiss.rev_swig_ptr(
        invlists.get_codes(0), list_size * invlists.code_size
    ).copy()
    packed = packed.reshape(list_size, invlists.code_size)
    stored_ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy().astype(np.int64)

    order = np.argsort(stored_ids, kind="stable")
    sorted_ids = stored_ids[order]
    expected = np.arange(n, dtype=np.int64)
    if not np.array_equal(sorted_ids, expected):
        raise RuntimeError(
            "FAISS stored IDs are not exactly 0..N-1; cannot safely align codes to items"
        )
    packed = packed[order]
    codes = _decode_packed_codes(packed, n_codebook, bits)

    if codes.min() < 0 or codes.max() >= codebook_size:
        raise RuntimeError(
            f"Decoded OPQ code out of range: min={codes.min()}, max={codes.max()}"
        )
    LOGGER.info(
        "OPQ complete: per-layer usage=%s",
        [int(len(np.unique(codes[:, i]))) for i in range(n_codebook)],
    )
    return codes, index, factory


def compute_sid_metrics(codes: np.ndarray, codebook_size: int) -> SIDMetrics:
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got {codes.shape}")
    tuples = [tuple(int(v) for v in row) for row in codes]
    counts = Counter(tuples)
    unique_full = len(counts)
    singleton_items = sum(count for count in counts.values() if count == 1)
    num_items = len(tuples)
    collision_items = num_items - singleton_items
    per_layer_unique = [int(len(np.unique(codes[:, i]))) for i in range(codes.shape[1])]
    return SIDMetrics(
        num_items=num_items,
        sid_length=int(codes.shape[1]),
        codebook_size=codebook_size,
        unique_full_sids=unique_full,
        collision_items=collision_items,
        collision_rate=float(1.0 - unique_full / max(num_items, 1)),
        singleton_items=singleton_items,
        icr=float(singleton_items / max(num_items, 1)),
        per_layer_unique=per_layer_unique,
        per_layer_cur=[float(v / codebook_size) for v in per_layer_unique],
        min_code=int(codes.min()),
        max_code=int(codes.max()),
    )


def append_collision_token(
    base_codes: np.ndarray,
    item_ids: np.ndarray,
    codebook_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Append a deterministic token that makes every full SID unique.

    Items sharing the same three-token base SID are ordered by ItemID and assigned
    collision ranks 0, 1, ..., group_size-1. Singleton groups receive token 0.
    """
    if base_codes.ndim != 2:
        raise ValueError(f"base_codes must be 2-D, got {base_codes.shape}")
    if len(base_codes) != len(item_ids):
        raise ValueError("base_codes and item_ids must have the same number of rows")

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for row_idx, row in enumerate(base_codes):
        groups.setdefault(tuple(int(v) for v in row), []).append(row_idx)

    collision_token = np.zeros(len(base_codes), dtype=np.int64)
    num_collision_groups = 0
    collision_items = 0
    max_collision_group_size = 1
    for indices in groups.values():
        size = len(indices)
        max_collision_group_size = max(max_collision_group_size, size)
        if size <= 1:
            continue
        num_collision_groups += 1
        collision_items += size
        ordered = sorted(indices, key=lambda idx: int(item_ids[idx]))
        for rank, row_idx in enumerate(ordered):
            collision_token[row_idx] = rank

    if max_collision_group_size > codebook_size:
        raise ValueError(
            "A collision group is larger than the collision-token codebook: "
            f"max_group={max_collision_group_size}, codebook_size={codebook_size}."
        )

    resolved = np.column_stack([base_codes, collision_token]).astype(np.int64)
    if len({tuple(int(v) for v in row) for row in resolved}) != len(resolved):
        raise RuntimeError("Collision token failed to make all SIDs unique")

    stats = {
        "num_collision_groups": int(num_collision_groups),
        "collision_items_before_resolution": int(collision_items),
        "max_collision_group_size": int(max_collision_group_size),
        "collision_token_unique": int(len(np.unique(collision_token))),
        "collision_token_max": int(collision_token.max(initial=0)),
    }
    return resolved, collision_token, stats


def save_outputs(
    output_dir: Path,
    dataset_name: str,
    method: str,
    item_ids: np.ndarray,
    base_codes: np.ndarray,
    codes: np.ndarray,
    base_metrics: SIDMetrics,
    metrics: SIDMetrics,
    metadata: Dict[str, object],
    rq_centroids: np.ndarray | None,
    rq_index: object | None,
    opq_index: object,
    rq_centroids_l1: np.ndarray | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / f"{dataset_name}_t5_{method}.npy"
    raw3_path = output_dir / f"{dataset_name}_t5_{method}_raw3.npy"
    np.save(code_path, codes.astype(np.int64))
    np.save(raw3_path, base_codes.astype(np.int64))

    mapping = pd.DataFrame({"ItemID": item_ids})
    for i in range(base_codes.shape[1]):
        mapping[f"base_sid_{i + 1}"] = base_codes[:, i]
    for i in range(codes.shape[1]):
        mapping[f"sid_{i + 1}"] = codes[:, i]
    if codes.shape[1] > base_codes.shape[1]:
        mapping["collision_token"] = codes[:, -1]
    mapping.to_csv(output_dir / f"{dataset_name}_t5_{method}_mapping.csv", index=False)

    if rq_centroids is not None:
        np.save(output_dir / "rq_centroids.npy", rq_centroids)
    if rq_centroids_l1 is not None:
        np.save(output_dir / "rq_centroids_level1.npy", rq_centroids_l1)
    if rq_index is not None:
        faiss.write_index(rq_index, str(output_dir / "rq_kmeans.index"))
    faiss.write_index(opq_index, str(output_dir / "opq.index"))

    payload = dict(metadata)
    payload["base_metrics_before_collision_resolution"] = asdict(base_metrics)
    payload["metrics"] = asdict(metrics)
    payload["code_path"] = str(code_path)
    payload["raw3_code_path"] = str(raw3_path)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved final SID array: %s shape=%s", code_path, codes.shape)
    LOGGER.info("Saved raw 3-token SID: %s shape=%s", raw3_path, base_codes.shape)
    LOGGER.info(
        "Before resolution: unique=%d/%d collision_rate=%.6f ICR=%.6f",
        base_metrics.unique_full_sids,
        base_metrics.num_items,
        base_metrics.collision_rate,
        base_metrics.icr,
    )
    LOGGER.info(
        "After resolution: unique=%d/%d collision_rate=%.6f ICR=%.6f layer_usage=%s",
        metrics.unique_full_sids,
        metrics.num_items,
        metrics.collision_rate,
        metrics.icr,
        metrics.per_layer_unique,
    )
    return code_path


def generate(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    parquet_path = dataset_dir / args.embedding_file
    item_ids, x, item_id_col, embedding_col = load_item_embeddings(
        parquet_path, args.item_id_col, args.embedding_col
    )

    if args.normalize_input:
        LOGGER.info("L2-normalizing input embeddings")
        x = maybe_l2_normalize(x)

    rq_centroids = None
    rq_index = None
    rq_centroids_l1 = None
    if args.method == "rq1_opq2":
        rq_codes, rq_centroids, opq_input, rq_index = train_rq_kmeans(
            x=x,
            codebook_size=args.codebook_size,
            niter=args.rq_niter,
            seed=args.seed,
            normalize_residual=args.normalize_residual,
        )
        opq_codes, opq_index, factory = train_and_encode_opq_source_style(
            opq_input,
            n_codebook=2,
            codebook_size=args.codebook_size,
        )
        base_codes = np.column_stack([rq_codes, opq_codes]).astype(np.int64)
    elif args.method == "rq2_opq3":
        # PATCH(rq2_opq3): 2 RQ tokens on the original embedding, then 3 OPQ/PQ
        # tokens on the level-2 residual. 768 % 3 == 0, so
        # "OPQ3,IVF1,PQ3x8" is constructible (unlike the impossible OPQ5).
        rq_codes, rq_centroids, opq_input, rq_index_pair, rq_centroids_l1 = train_rq_kmeans_2level(
            x=x,
            codebook_size=args.codebook_size,
            niter=args.rq_niter,
            seed=args.seed,
            normalize_residual=args.normalize_residual,
        )
        opq_codes, opq_index, factory = train_and_encode_opq_source_style(
            opq_input,
            n_codebook=3,
            codebook_size=args.codebook_size,
        )
        base_codes = np.column_stack([rq_codes, opq_codes]).astype(np.int64)
        # save_outputs() takes a single faiss index; hand it the LEVEL-2 index,
        # which is the one whose residual the OPQ stage consumed. The level-1
        # centroids are persisted separately via rq_centroids_l1.
        rq_index = rq_index_pair[1]
    elif args.method == "opq3":
        opq_codes, opq_index, factory = train_and_encode_opq_source_style(
            x,
            n_codebook=3,
            codebook_size=args.codebook_size,
        )
        base_codes = opq_codes.astype(np.int64)
    else:  # guarded by argparse
        raise ValueError(f"Unknown method: {args.method}")

    # PATCH(rq2_opq3): the historical code hard-coded 3 semantic positions.
    # The width is now derived from the method instead of assumed.
    expected_width = 5 if args.method == "rq2_opq3" else 3
    if base_codes.shape != (len(item_ids), expected_width):
        raise RuntimeError(
            f"Expected base SID shape {(len(item_ids), expected_width)}, "
            f"got {base_codes.shape}"
        )
    if base_codes.min() < 0 or base_codes.max() >= args.codebook_size:
        raise RuntimeError(
            f"Base SID codes must be in [0,{args.codebook_size - 1}], "
            f"got [{base_codes.min()},{base_codes.max()}]"
        )

    base_metrics = compute_sid_metrics(base_codes, args.codebook_size)
    collision_stats: Dict[str, int] = {}
    if args.add_collision_token:
        codes, _, collision_stats = append_collision_token(
            base_codes=base_codes,
            item_ids=item_ids,
            codebook_size=args.codebook_size,
        )
    else:
        codes = base_codes
    metrics = compute_sid_metrics(codes, args.codebook_size)
    dataset_name = dataset_dir.name
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        dataset_dir / "rq_opq" / args.method
    )

    metadata: Dict[str, object] = {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "embedding_file": str(parquet_path),
        "item_id_col": item_id_col,
        "embedding_col": embedding_col,
        "embedding_shape": list(x.shape),
        "method": args.method,
        "description": (
            "1 RQ KMeans token + 2 OPQ residual tokens"
            if args.method == "rq1_opq2"
            else ("2 RQ KMeans tokens + 3 OPQ residual tokens  [PATCH(rq2_opq3)]"
                  if args.method == "rq2_opq3"
                  else "3 OPQ tokens on original embedding")
        ),
        "rq_levels": 2 if args.method == "rq2_opq3" else (1 if args.method == "rq1_opq2" else 0),
        "source_template": ("the original RQ-OPQ generator implementation (not shipped); "
                            "generate_rq_opq_sid.py"),
        "patched_by": "outputs/final_hyperparam_efficiency/drivers (rq2_opq3 addition only)",
        "codebook_size": args.codebook_size,
        "rq_niter": args.rq_niter,
        "seed": args.seed,
        "normalize_input": args.normalize_input,
        "normalize_residual": args.normalize_residual,
        "opq_factory": factory,
        "add_collision_token": args.add_collision_token,
        "collision_resolution": collision_stats,
        "final_sid_length": int(codes.shape[1]),
        "tiger_note": (
            "Output contains raw 0-based codes; the isolated adapter adds per-layer offsets."
        ),
    }
    return save_outputs(
        output_dir=output_dir,
        dataset_name=dataset_name,
        method=args.method,
        item_ids=item_ids,
        base_codes=base_codes,
        codes=codes,
        base_metrics=base_metrics,
        metrics=metrics,
        metadata=metadata,
        rq_centroids=rq_centroids,
        rq_index=rq_index,
        opq_index=opq_index,
        rq_centroids_l1=rq_centroids_l1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OneSearch-style RQ-OPQ or pure OPQ SIDs for TIGER"
    )
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument(
        # PATCH(rq2_opq3): the historical enum had only ["rq1_opq2", "opq3"],
        # both 3 semantic positions. "rq2_opq3" is added; nothing else changes.
        "--method", required=True, choices=["rq1_opq2", "opq3", "rq2_opq3"]
    )
    parser.add_argument(
        "--embedding_file", default="item_emb.parquet", help="File inside dataset_dir"
    )
    parser.add_argument("--item_id_col", default=None)
    parser.add_argument("--embedding_col", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--rq_niter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--normalize_input",
        action="store_true",
        help="Optional L2 normalization before all quantization (off by default)",
    )
    parser.add_argument(
        "--normalize_residual",
        action="store_true",
        help="Optional L2 normalization after RQ before OPQ (off by default)",
    )
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--add_collision_token",
        dest="add_collision_token",
        action="store_true",
        help="Append a fourth token that uniquely ranks items inside each collision group",
    )
    collision_group.add_argument(
        "--no_collision_token",
        dest="add_collision_token",
        action="store_false",
        help="Keep only the original three quantization tokens",
    )
    parser.set_defaults(add_collision_token=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    code_path = generate(args)
    print(code_path)


if __name__ == "__main__":
    main()
