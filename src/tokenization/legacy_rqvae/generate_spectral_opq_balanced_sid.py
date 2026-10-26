import os
import argparse
import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA


def load_item_embedding(path):
    df = pd.read_parquet(path)

    print("Loaded parquet columns:", df.columns.tolist())
    print("Data shape:", df.shape)

    possible_emb_cols = ["embedding", "emb", "item_emb", "vector"]

    for col in possible_emb_cols:
        if col in df.columns:
            X = np.stack(df[col].values).astype(np.float32)
            return X, df

    if df.shape[1] == 1:
        first_col = df.columns[0]
        first_val = df[first_col].iloc[0]
        if isinstance(first_val, (list, tuple, np.ndarray)):
            X = np.stack(df[first_col].values).astype(np.float32)
            return X, df

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 0:
        X = numeric_df.values.astype(np.float32)
        return X, df

    raise ValueError("could not identify an embedding column in the parquet file.")


def apply_opq_or_pca_whitening(X, num_groups=3, opq_niter=50, random_state=2025):
    """
    Prefer faiss.OPQMatrix.
    If faiss is unavailable, fall back to PCA whitening as the rotation.
    """
    X = X.astype(np.float32)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    print("=" * 80)
    print("Step 0: OPQ rotation / PCA whitening")

    try:
        import faiss

        print("faiss detected. Using faiss.OPQMatrix...")
        d = X.shape[1]

        opq = faiss.OPQMatrix(d, num_groups)
        opq.niter = opq_niter
        opq.niter_pq = 20
        opq.niter_pq_0 = 40
        opq.verbose = True

        opq.train(X)
        X_rot = opq.apply_py(X).astype(np.float32)

        print("OPQ rotation finished.")
        return X_rot, "faiss_opq"

    except Exception as e:
        print("faiss OPQ unavailable or failed.")
        print("Fallback to PCA whitening.")
        print("Reason:", repr(e))

        pca = PCA(
            n_components=X.shape[1],
            whiten=True,
            random_state=random_state,
        )
        X_rot = pca.fit_transform(X).astype(np.float32)

        print("PCA whitening finished.")
        return X_rot, "pca_whitening"


def build_dimension_affinity(X, metric="abs_pearson", eps=1e-8):
    print("=" * 80)
    print("Building dimension affinity matrix...")

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


def spectral_group_dimensions(X, num_groups=3, metric="abs_pearson", random_state=2025):
    num_items, emb_dim = X.shape
    print("=" * 80)
    print(f"Spectral clustering dimensions into {num_groups} groups...")
    print(f"Embedding matrix shape: num_items={num_items}, emb_dim={emb_dim}")

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
        print(f"Group {g}: {len(idx)} dimensions")

    return groups, dim_labels


def make_quotas(n, k, natural_labels=None):
    """
    Build the capacity constraint of balanced k-means.
    Each code receives about n/k items.
    """
    base = n // k
    rem = n % k

    quotas = np.full(k, base, dtype=np.int64)

    if rem > 0:
        if natural_labels is not None:
            counts = np.bincount(natural_labels, minlength=k)
            order = np.argsort(-counts)
            plus_centers = order[:rem]
        else:
            plus_centers = np.arange(rem)

        quotas[plus_centers] += 1

    assert quotas.sum() == n
    return quotas


def balanced_assignment(distances, quotas):
    """
    Approximate balanced assignment from item-center distances under the capacity constraint.

    distances: [num_items, num_clusters]
    quotas: [num_clusters], sum = num_items
    """
    n, k = distances.shape

    labels = np.full(n, -1, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)

    flat_order = np.argsort(distances.reshape(-1))

    assigned = 0

    for flat_idx in flat_order:
        i = flat_idx // k
        c = flat_idx % k

        if labels[i] != -1:
            continue

        if counts[c] >= quotas[c]:
            continue

        labels[i] = c
        counts[c] += 1
        assigned += 1

        if assigned == n:
            break

    if assigned < n:
        remaining_items = np.where(labels == -1)[0]
        remaining_centers = []
        for c in range(k):
            left = quotas[c] - counts[c]
            if left > 0:
                remaining_centers.extend([c] * int(left))

        remaining_centers = np.array(remaining_centers, dtype=np.int64)

        for item_idx, c in zip(remaining_items, remaining_centers):
            labels[item_idx] = c
            counts[c] += 1

    assert np.all(labels >= 0)
    assert counts.sum() == n

    return labels


def balanced_kmeans(
    X,
    n_clusters=256,
    n_iter=8,
    random_state=2025,
):
    """
    Approximate balanced k-means:
    1. initialise centers with ordinary k-means;
    2. reassign items under the capacity constraint;
    3. recompute centers;
    4. iterate a few rounds.
    """
    n, d = X.shape

    print(f"Running balanced k-means: n={n}, dim={d}, clusters={n_clusters}")

    init_kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=300,
        verbose=0,
    )
    natural_labels = init_kmeans.fit_predict(X)
    centers = init_kmeans.cluster_centers_.astype(np.float32)

    quotas = make_quotas(n, n_clusters, natural_labels=natural_labels)

    for it in range(n_iter):
        print(f"Balanced k-means iter {it + 1}/{n_iter}")

        # distances: [n, k]
        x_norm = np.sum(X * X, axis=1, keepdims=True)
        c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
        distances = x_norm + c_norm - 2.0 * (X @ centers.T)
        distances = np.maximum(distances, 0.0)

        labels = balanced_assignment(distances, quotas)

        new_centers = np.zeros_like(centers)
        for c in range(n_clusters):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                new_centers[c] = X[idx].mean(axis=0)
            else:
                new_centers[c] = centers[c]

        shift = np.linalg.norm(new_centers - centers)
        print(f"Center shift: {shift:.6f}")

        centers = new_centers

    # recompute distances once and return the final labels and distance matrix
    x_norm = np.sum(X * X, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    distances = x_norm + c_norm - 2.0 * (X @ centers.T)
    distances = np.maximum(distances, 0.0).astype(np.float32)

    labels = balanced_assignment(distances, quotas)

    counts = np.bincount(labels, minlength=n_clusters)
    print("Code usage min/max:", counts.min(), counts.max())
    print("Code usage mean:", counts.mean())

    return labels, centers, distances


def reduce_if_needed(X_sub, pca_dim=64, random_state=2025):
    if pca_dim is None or pca_dim <= 0:
        return X_sub

    if X_sub.shape[1] <= pca_dim:
        return X_sub

    pca = PCA(
        n_components=pca_dim,
        whiten=True,
        random_state=random_state,
    )
    return pca.fit_transform(X_sub).astype(np.float32)


def generate_sid_by_balanced_kmeans(
    X,
    groups,
    codebook_size=256,
    pca_dim=64,
    balanced_iter=8,
    random_state=2025,
):
    num_items = X.shape[0]
    num_groups = len(groups)

    codes = np.zeros((num_items, num_groups), dtype=np.int64)
    all_distances = []

    for g, dims in enumerate(groups):
        print("=" * 80)
        print(f"Generating SID token {g + 1}/{num_groups}")
        print(f"Subspace dim: {len(dims)}")

        X_sub = X[:, dims]
        X_sub = StandardScaler().fit_transform(X_sub).astype(np.float32)
        X_sub = reduce_if_needed(
            X_sub,
            pca_dim=pca_dim,
            random_state=random_state,
        )

        print(f"Balanced k-means input shape: {X_sub.shape}")

        labels, centers, distances = balanced_kmeans(
            X_sub,
            n_clusters=codebook_size,
            n_iter=balanced_iter,
            random_state=random_state,
        )

        codes[:, g] = labels
        all_distances.append(distances)

        unique_num = len(np.unique(labels))
        print(f"Unique codes in token {g}: {unique_num}/{codebook_size}")

    return codes, all_distances


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

        print(f"Token {g}: top candidate shape = {idx.shape}")

    return top_indices, top_costs


def resolve_collisions_by_reassignment(
    codes,
    all_distances,
    top_m=64,
    max_combinations=20000,
):
    """
    Reassign collisions without changing the number of SID positions.
    """
    print("=" * 80)
    print("Resolving collisions without adding new SID token...")

    num_items, num_tokens = codes.shape
    top_indices, top_costs = get_top_candidates(all_distances, top_m=top_m)

    used = set()
    duplicate_items = []

    for i in range(num_items):
        sid = tuple(codes[i].tolist())
        if sid not in used:
            used.add(sid)
        else:
            duplicate_items.append(i)

    print(f"Initial duplicate items needing reassignment: {len(duplicate_items)}")

    reassigned = 0
    failed = 0

    for count, item_idx in enumerate(duplicate_items, start=1):
        if count % 500 == 0:
            print(f"Processed {count}/{len(duplicate_items)}, reassigned={reassigned}, failed={failed}")

        candidate_lists = []
        cost_lists = []

        for g in range(num_tokens):
            candidate_lists.append(top_indices[g][item_idx])
            cost_lists.append(top_costs[g][item_idx])

        best_sid = None
        best_cost = float("inf")

        checked = 0

        for combo_pos in itertools.product(range(top_m), repeat=num_tokens):
            checked += 1
            if checked > max_combinations:
                break

            sid = tuple(
                int(candidate_lists[g][combo_pos[g]])
                for g in range(num_tokens)
            )

            if sid in used:
                continue

            cost = sum(
                float(cost_lists[g][combo_pos[g]])
                for g in range(num_tokens)
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

    print(f"Reassigned duplicate items: {reassigned}")
    print(f"Failed to reassign: {failed}")

    return codes


def code_usage_entropy(codes, codebook_size):
    """
    Compute the usage entropy of the codes at each position.
    Higher entropy means the codes are used more uniformly.
    """
    entropies = []

    for g in range(codes.shape[1]):
        counts = np.bincount(codes[:, g], minlength=codebook_size).astype(np.float64)
        prob = counts / counts.sum()
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob))
        entropy_norm = entropy / np.log(codebook_size)
        entropies.append(entropy_norm)

    return entropies


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--item_emb_path", type=str, default="../data/Beauty/item_emb.parquet")
    parser.add_argument("--output_path", type=str, default="../data/Beauty/Beauty_t5_spectral_opq_balanced.npy")

    parser.add_argument("--num_sid_tokens", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)

    parser.add_argument("--metric", type=str, default="abs_pearson", choices=["pearson", "abs_pearson", "cosine"])
    parser.add_argument("--pca_dim", type=int, default=64)

    parser.add_argument("--opq_niter", type=int, default=50)
    parser.add_argument("--balanced_iter", type=int, default=8)

    parser.add_argument("--top_m", type=int, default=64)
    parser.add_argument("--max_combinations", type=int, default=20000)

    parser.add_argument("--random_state", type=int, default=2025)

    args = parser.parse_args()

    print("Loading item embeddings...")
    X, _ = load_item_embedding(args.item_emb_path)
    print("Original embedding shape:", X.shape)

    X_rot, rotate_method = apply_opq_or_pca_whitening(
        X,
        num_groups=args.num_sid_tokens,
        opq_niter=args.opq_niter,
        random_state=args.random_state,
    )

    print("Rotation method:", rotate_method)
    print("Rotated embedding shape:", X_rot.shape)

    print("=" * 80)
    print("Step 1: spectral clustering on rotated embedding dimensions...")
    groups, dim_labels = spectral_group_dimensions(
        X_rot,
        num_groups=args.num_sid_tokens,
        metric=args.metric,
        random_state=args.random_state,
    )

    pca_dim = None if args.pca_dim <= 0 else args.pca_dim

    print("=" * 80)
    print("Step 2: group-wise balanced k-means SID generation...")
    codes, all_distances = generate_sid_by_balanced_kmeans(
        X_rot,
        groups,
        codebook_size=args.codebook_size,
        pca_dim=pca_dim,
        balanced_iter=args.balanced_iter,
        random_state=args.random_state,
    )

    before_rate, before_unique, total_num = compute_collision_rate(codes)

    print("=" * 80)
    print("Before reassignment:")
    print(f"Total items: {total_num}")
    print(f"Unique SID: {before_unique}")
    print(f"Collision rate: {before_rate:.6f}")

    entropies = code_usage_entropy(codes, args.codebook_size)
    for i, e in enumerate(entropies):
        print(f"Token {i} usage entropy: {e:.6f}")

    print("=" * 80)
    print("Step 3: collision-aware reassignment...")
    codes = resolve_collisions_by_reassignment(
        codes,
        all_distances,
        top_m=args.top_m,
        max_combinations=args.max_combinations,
    )

    after_rate, after_unique, total_num = compute_collision_rate(codes)

    print("=" * 80)
    print("After reassignment:")
    print(f"Total items: {total_num}")
    print(f"Unique SID: {after_unique}")
    print(f"Collision rate: {after_rate:.6f}")

    entropies = code_usage_entropy(codes, args.codebook_size)
    for i, e in enumerate(entropies):
        print(f"Token {i} usage entropy after reassignment: {e:.6f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.save(args.output_path, codes)

    group_path = args.output_path.replace(".npy", "_dim_groups.npy")
    label_path = args.output_path.replace(".npy", "_dim_labels.npy")
    rotate_info_path = args.output_path.replace(".npy", "_rotate_method.txt")

    np.save(group_path, np.array(groups, dtype=object), allow_pickle=True)
    np.save(label_path, dim_labels)

    with open(rotate_info_path, "w") as f:
        f.write(rotate_method + "\n")

    print("=" * 80)
    print(f"Saved SID to: {args.output_path}")
    print(f"Saved dimension groups to: {group_path}")
    print(f"Saved dimension labels to: {label_path}")
    print(f"Saved rotate method to: {rotate_info_path}")


if __name__ == "__main__":
    main()
