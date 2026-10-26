import sys as _sys
from pathlib import Path as _Path
# local sibling modules of the reference RQ-VAE implementation
_SIB = str(_Path(__file__).resolve().parent)
if _SIB not in _sys.path:
    _sys.path.insert(0, _SIB)
import os
import argparse
import itertools
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA

from datasets import EmbDataset
from models.rqvae import RQVAE


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_rqvae_from_ckpt(ckpt, data_dim):
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    model = RQVAE(
        in_dim=data_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=getattr(args, "beta", 0.25),
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict)
    return model, args


@torch.no_grad()
def extract_first_rq_residual(
    model,
    dataset,
    device="cuda:0",
    batch_size=256,
    num_workers=4,
):
    """
    Output:
    first_codes: [num_items]
    residuals:   [num_items, e_dim]

    Logic:
    x -> encoder -> z
    z -> first RQ layer -> q1, c1
    residual = z - q1
    """
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_first_codes = []
    all_residuals = []
    all_z = []

    first_vq = model.rq.vq_layers[0]

    for x in tqdm(loader, desc="Extracting first RQ residual"):
        x = x.to(device)

        z = model.encoder(x)

        q1, _, c1 = first_vq(z, use_sk=False)

        residual = z - q1

        all_first_codes.append(c1.view(-1).cpu().numpy())
        all_residuals.append(residual.cpu().numpy())
        all_z.append(z.cpu().numpy())

    first_codes = np.concatenate(all_first_codes, axis=0).astype(np.int64)
    residuals = np.concatenate(all_residuals, axis=0).astype(np.float32)
    z_all = np.concatenate(all_z, axis=0).astype(np.float32)

    return first_codes, residuals, z_all


def build_dimension_affinity(X, metric="abs_pearson", eps=1e-8):
    """
    Build the similarity matrix over the dimensions of the residual embedding.
    X: [num_items, residual_dim]
    """
    X_std = StandardScaler().fit_transform(X)

    if metric == "pearson":
        S = np.corrcoef(X_std.T)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = (S + 1.0) / 2.0

    elif metric == "abs_pearson":
        S = np.corrcoef(X_std.T)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.abs(S)

    elif metric == "cosine":
        V = X_std.T
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)
        S = V @ V.T
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = (S + 1.0) / 2.0

    else:
        raise ValueError(f"Unknown metric: {metric}")

    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)


def spectral_group_dimensions(
    X,
    num_groups=2,
    metric="abs_pearson",
    random_state=2025,
):
    print("=" * 80)
    print(f"Spectral clustering residual dimensions into {num_groups} groups")
    print("Residual matrix shape:", X.shape)

    S = build_dimension_affinity(X, metric=metric)

    clustering = SpectralClustering(
        n_clusters=num_groups,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=20,
    )

    dim_labels = clustering.fit_predict(S)

    groups = []
    for g in range(num_groups):
        idx = np.where(dim_labels == g)[0]
        groups.append(idx)
        print(f"Residual dimension group {g}: {len(idx)} dims")

    return groups, dim_labels


def reduce_if_needed(X_sub, pca_dim=0, random_state=2025):
    if pca_dim is None or pca_dim <= 0:
        return X_sub

    if X_sub.shape[1] <= pca_dim:
        return X_sub

    pca = PCA(
        n_components=pca_dim,
        whiten=True,
        random_state=random_state,
    )
    return pca.fit_transform(X_sub)


def generate_spectral_tokens_from_residual(
    residuals,
    groups,
    codebook_size=256,
    pca_dim=0,
    random_state=2025,
):
    """
    Run k-means on each spectral dimension group of the residual.
    Output spectral_codes: [num_items, num_groups]
    """
    num_items = residuals.shape[0]
    num_groups = len(groups)

    spectral_codes = np.zeros((num_items, num_groups), dtype=np.int64)
    all_distances = []

    for g, dims in enumerate(groups):
        print("=" * 80)
        print(f"Generating residual spectral token {g + 1}/{num_groups}")
        print(f"Subspace dim: {len(dims)}")

        X_sub = residuals[:, dims]
        X_sub = StandardScaler().fit_transform(X_sub)

        X_sub = reduce_if_needed(
            X_sub,
            pca_dim=pca_dim,
            random_state=random_state,
        )

        print("KMeans input shape:", X_sub.shape)

        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=random_state + g,
            n_init=30,
            max_iter=500,
            verbose=0,
        )

        labels = kmeans.fit_predict(X_sub)
        distances = kmeans.transform(X_sub).astype(np.float32)

        spectral_codes[:, g] = labels
        all_distances.append(distances)

        print(f"Used codes: {len(np.unique(labels))}/{codebook_size}")

    return spectral_codes, all_distances


def compute_collision_rate(codes):
    sid_tuples = [tuple(row.tolist()) for row in codes]
    unique_num = len(set(sid_tuples))
    total_num = len(sid_tuples)
    collision_rate = 1.0 - unique_num / total_num
    return collision_rate, unique_num, total_num


def get_top_candidates(all_distances, top_m=64):
    top_indices = []
    top_costs = []

    for g, dist in enumerate(all_distances):
        idx = np.argsort(dist, axis=1)[:, :top_m]
        cost = np.take_along_axis(dist, idx, axis=1)

        top_indices.append(idx)
        top_costs.append(cost)

        print(f"Spectral token {g}: top candidate shape = {idx.shape}")

    return top_indices, top_costs


def resolve_collisions_keep_rq_fixed(
    codes,
    all_distances,
    top_m=64,
    max_combinations=4096,
):
    """
    Keep the first RQ token fixed and reassign only the later spectral tokens.
    codes: [num_items, 3]
    all_distances: distance lists of the residual spectral tokens, length 2.
    """
    print("=" * 80)
    print("Resolving collisions while keeping RQ token fixed")

    num_items, num_tokens = codes.shape
    num_spectral_tokens = num_tokens - 1

    top_indices, top_costs = get_top_candidates(all_distances, top_m=top_m)

    used = set()
    duplicate_items = []

    for i in range(num_items):
        sid = tuple(codes[i].tolist())
        if sid not in used:
            used.add(sid)
        else:
            duplicate_items.append(i)

    print(f"Duplicate items needing reassignment: {len(duplicate_items)}")

    reassigned = 0
    failed = 0

    for count, item_idx in enumerate(duplicate_items, start=1):
        if count % 500 == 0:
            print(
                f"Processed {count}/{len(duplicate_items)}, "
                f"reassigned={reassigned}, failed={failed}"
            )

        rq_code = int(codes[item_idx, 0])

        candidate_lists = []
        cost_lists = []

        for g in range(num_spectral_tokens):
            candidate_lists.append(top_indices[g][item_idx])
            cost_lists.append(top_costs[g][item_idx])

        best_sid = None
        best_cost = float("inf")

        checked = 0

        for combo_pos in itertools.product(range(top_m), repeat=num_spectral_tokens):
            checked += 1
            if checked > max_combinations:
                break

            spectral_part = [
                int(candidate_lists[g][combo_pos[g]])
                for g in range(num_spectral_tokens)
            ]

            sid = tuple([rq_code] + spectral_part)

            if sid in used:
                continue

            cost = sum(
                float(cost_lists[g][combo_pos[g]])
                for g in range(num_spectral_tokens)
            )

            if cost < best_cost:
                best_cost = cost
                best_sid = sid

        if best_sid is not None:
            codes[item_idx] = np.array(best_sid, dtype=np.int64)
            used.add(best_sid)
            reassigned += 1
        else:
            failed += 1

    print(f"Reassigned: {reassigned}")
    print(f"Failed: {failed}")

    return codes


def print_code_usage(codes, codebook_size):
    print("=" * 80)
    print("Code usage statistics")

    for g in range(codes.shape[1]):
        labels = codes[:, g]
        counts = np.bincount(labels, minlength=codebook_size)
        used = np.sum(counts > 0)

        probs = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        norm_entropy = entropy / np.log(codebook_size)

        print(
            f"Token {g}: used={used}/{codebook_size}, "
            f"min_count={counts[counts > 0].min()}, "
            f"max_count={counts.max()}, "
            f"norm_entropy={norm_entropy:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--item_emb_path",
        type=str,
        default="../data/Beauty/item_emb.parquet",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/Beauty/Beauty_t5_rq_residual_spectral.npy",
    )

    parser.add_argument("--num_sid_tokens", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument(
        "--metric",
        type=str,
        default="abs_pearson",
        choices=["pearson", "abs_pearson", "cosine"],
    )
    parser.add_argument("--pca_dim", type=int, default=0)
    parser.add_argument("--top_m", type=int, default=64)
    parser.add_argument("--max_combinations", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random_state", type=int, default=2025)
    parser.add_argument("--no_reassign", action="store_true")

    args = parser.parse_args()

    assert args.num_sid_tokens >= 2, "num_sid_tokens must be at least 2"
    spectral_token_num = args.num_sid_tokens - 1

    print("=" * 80)
    print("Loading item embeddings")
    dataset = EmbDataset(args.item_emb_path)
    print("Number of items:", len(dataset))
    print("Input embedding dim:", dataset.dim)

    print("=" * 80)
    print("Loading RQVAE checkpoint")
    ckpt = safe_torch_load(args.ckpt_path, map_location="cpu")
    model, rq_args = build_rqvae_from_ckpt(ckpt, data_dim=dataset.dim)

    print("RQVAE num_emb_list:", rq_args.num_emb_list)
    print("RQVAE e_dim:", rq_args.e_dim)

    if args.codebook_size != rq_args.num_emb_list[0]:
        print(
            "Warning: codebook_size differs from the RQ first-layer codebook size."
            f" current codebook_size={args.codebook_size}, "
            f"RQ first codebook={rq_args.num_emb_list[0]}"
        )

    print("=" * 80)
    print("Step 1: extract first RQ token and first residual")
    first_codes, residuals, z_all = extract_first_rq_residual(
        model=model,
        dataset=dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("First RQ codes shape:", first_codes.shape)
    print("Residuals shape:", residuals.shape)
    print("Latent z shape:", z_all.shape)

    print("=" * 80)
    print("Step 2: spectral clustering on residual dimensions")
    groups, dim_labels = spectral_group_dimensions(
        residuals,
        num_groups=spectral_token_num,
        metric=args.metric,
        random_state=args.random_state,
    )

    print("=" * 80)
    print("Step 3: group-wise k-means on residual subspaces")
    spectral_codes, all_distances = generate_spectral_tokens_from_residual(
        residuals=residuals,
        groups=groups,
        codebook_size=args.codebook_size,
        pca_dim=args.pca_dim,
        random_state=args.random_state,
    )

    codes = np.concatenate(
        [first_codes.reshape(-1, 1), spectral_codes],
        axis=1,
    ).astype(np.int64)

    before_rate, before_unique, total_num = compute_collision_rate(codes)

    print("=" * 80)
    print("Before reassignment:")
    print(f"Total items: {total_num}")
    print(f"Unique SID: {before_unique}")
    print(f"Collision rate: {before_rate:.6f}")

    if not args.no_reassign:
        codes = resolve_collisions_keep_rq_fixed(
            codes=codes,
            all_distances=all_distances,
            top_m=args.top_m,
            max_combinations=args.max_combinations,
        )

    after_rate, after_unique, total_num = compute_collision_rate(codes)

    print("=" * 80)
    print("After reassignment:")
    print(f"Total items: {total_num}")
    print(f"Unique SID: {after_unique}")
    print(f"Collision rate: {after_rate:.6f}")

    print_code_usage(codes, args.codebook_size)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.save(args.output_path, codes)

    group_path = args.output_path.replace(".npy", "_residual_dim_groups.npy")
    label_path = args.output_path.replace(".npy", "_residual_dim_labels.npy")
    residual_path = args.output_path.replace(".npy", "_residual.npy")
    z_path = args.output_path.replace(".npy", "_latent_z.npy")
    meta_path = args.output_path.replace(".npy", "_meta.txt")

    np.save(group_path, np.array(groups, dtype=object), allow_pickle=True)
    np.save(label_path, dim_labels)
    np.save(residual_path, residuals)
    np.save(z_path, z_all)

    with open(meta_path, "w") as f:
        f.write("method=RQ1 + Residual Spectral KMeans\n")
        f.write(f"ckpt_path={args.ckpt_path}\n")
        f.write(f"item_emb_path={args.item_emb_path}\n")
        f.write(f"num_sid_tokens={args.num_sid_tokens}\n")
        f.write(f"spectral_token_num={spectral_token_num}\n")
        f.write(f"codebook_size={args.codebook_size}\n")
        f.write(f"metric={args.metric}\n")
        f.write(f"pca_dim={args.pca_dim}\n")
        f.write(f"before_collision_rate={before_rate:.6f}\n")
        f.write(f"after_collision_rate={after_rate:.6f}\n")
        f.write(f"unique_sid={after_unique}\n")
        f.write(f"total_items={total_num}\n")

    print("=" * 80)
    print(f"Saved SID to: {args.output_path}")
    print(f"Saved residual dim groups to: {group_path}")
    print(f"Saved residual dim labels to: {label_path}")
    print(f"Saved residual matrix to: {residual_path}")
    print(f"Saved latent z to: {z_path}")
    print(f"Saved meta to: {meta_path}")


if __name__ == "__main__":
    main()
