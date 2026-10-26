import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_item_embedding(path):
    df = pd.read_parquet(path)

    possible_emb_cols = ["embedding", "emb", "item_emb", "vector"]

    for col in possible_emb_cols:
        if col in df.columns:
            X = np.stack(df[col].values).astype(np.float32)
            return X

    if df.shape[1] == 1:
        first_col = df.columns[0]
        first_val = df[first_col].iloc[0]
        if isinstance(first_val, (list, tuple, np.ndarray)):
            X = np.stack(df[first_col].values).astype(np.float32)
            return X

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 0:
        return numeric_df.values.astype(np.float32)

    raise ValueError("Cannot identify item embeddings from parquet file.")


def make_dimension_groups(emb_dim, num_groups, split_method="random", random_state=2025):
    dims = np.arange(emb_dim)

    if split_method == "random":
        rng = np.random.default_rng(random_state)
        rng.shuffle(dims)
        groups = np.array_split(dims, num_groups)

    elif split_method == "even":
        groups = np.array_split(dims, num_groups)

    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    groups = [np.asarray(g, dtype=np.int64) for g in groups]

    dim_labels = np.zeros(emb_dim, dtype=np.int64)
    for gid, g in enumerate(groups):
        dim_labels[g] = gid

    return groups, dim_labels


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
    return pca.fit_transform(X_sub)


def generate_base_sid_by_group_kmeans(
    X,
    groups,
    codebook_size=256,
    pca_dim=64,
    random_state=2025,
):
    num_items = X.shape[0]
    num_groups = len(groups)

    codes = np.zeros((num_items, num_groups), dtype=np.int64)

    for g, dims in enumerate(groups):
        print("=" * 80)
        print(f"Generating base SID token {g + 1}/{num_groups}")
        print(f"Subspace dim: {len(dims)}")

        X_sub = X[:, dims]
        X_sub = StandardScaler().fit_transform(X_sub)
        X_sub = reduce_if_needed(
            X_sub,
            pca_dim=pca_dim,
            random_state=random_state,
        )

        print(f"KMeans input shape: {X_sub.shape}")

        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=random_state,
            n_init=30,
            max_iter=500,
            verbose=0,
        )

        labels = kmeans.fit_predict(X_sub)
        codes[:, g] = labels

        unique_num = len(np.unique(labels))
        print(f"Unique codes in token {g}: {unique_num}/{codebook_size}")

    return codes


def compute_collision_rate(codes):
    sid_tuples = [tuple(row.tolist()) for row in codes]
    unique_num = len(set(sid_tuples))
    total_num = len(sid_tuples)
    collision_rate = 1.0 - unique_num / total_num
    return collision_rate, unique_num, total_num


def add_collision_token(codes, codebook_size=256):
    print("=" * 80)
    print("Adding collision-resolution token at the end...")

    num_items, num_tokens = codes.shape

    new_codes = np.zeros((num_items, num_tokens + 1), dtype=np.int64)
    new_codes[:, :num_tokens] = codes

    sid_to_items = {}

    for i in range(num_items):
        sid = tuple(codes[i].tolist())
        sid_to_items.setdefault(sid, []).append(i)

    duplicate_group_num = 0
    duplicate_item_num = 0
    max_collision_size = 1

    for sid, item_indices in sid_to_items.items():
        group_size = len(item_indices)

        if group_size > 1:
            duplicate_group_num += 1
            duplicate_item_num += group_size - 1
            max_collision_size = max(max_collision_size, group_size)

        for conflict_id, item_idx in enumerate(item_indices):
            new_codes[item_idx, -1] = conflict_id

    print(f"Original SID shape: {codes.shape}")
    print(f"New SID shape: {new_codes.shape}")
    print(f"Duplicate SID groups: {duplicate_group_num}")
    print(f"Duplicate items beyond first: {duplicate_item_num}")
    print(f"Max collision group size: {max_collision_size}")
    print(f"Max conflict token value: {int(new_codes[:, -1].max())}")

    if int(new_codes[:, -1].max()) >= codebook_size:
        raise RuntimeError(
            f"Conflict token max value {int(new_codes[:, -1].max())} >= codebook_size {codebook_size}. "
            "Current TIGER offset assumes each SID position has 256 ids."
        )

    return new_codes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--item_emb_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_sid_tokens", type=int, required=True)
    parser.add_argument("--split_method", type=str, required=True, choices=["random", "even"])
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=2025)

    args = parser.parse_args()

    print("=" * 80)
    print("Loading item embeddings...")
    X = load_item_embedding(args.item_emb_path)
    print("Embedding shape:", X.shape)

    pca_dim = None if args.pca_dim <= 0 else args.pca_dim

    print("=" * 80)
    print(f"Step 1: {args.split_method} dimension split...")
    groups, dim_labels = make_dimension_groups(
        emb_dim=X.shape[1],
        num_groups=args.num_sid_tokens,
        split_method=args.split_method,
        random_state=args.random_state,
    )

    for gid, g in enumerate(groups):
        print(f"Group {gid}: {len(g)} dimensions")

    print("=" * 80)
    print("Step 2: base group-wise KMeans SID generation...")
    base_codes = generate_base_sid_by_group_kmeans(
        X,
        groups,
        codebook_size=args.codebook_size,
        pca_dim=pca_dim,
        random_state=args.random_state,
    )

    before_rate, before_unique, total_num = compute_collision_rate(base_codes)

    print("=" * 80)
    print("Before adding collision token:")
    print(f"Total items: {total_num}")
    print(f"Unique base SID: {before_unique}")
    print(f"Base collision rate: {before_rate:.6f}")

    print("=" * 80)
    print("Step 3: add final collision-resolution token...")
    final_codes = add_collision_token(base_codes, codebook_size=args.codebook_size)

    after_rate, after_unique, total_num = compute_collision_rate(final_codes)

    print("=" * 80)
    print("After adding collision token:")
    print(f"Total items: {total_num}")
    print(f"Unique final SID: {after_unique}")
    print(f"Final collision rate: {after_rate:.6f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.save(args.output_path, final_codes)

    base_path = args.output_path.replace(".npy", "_base_sid.npy")
    group_path = args.output_path.replace(".npy", "_dim_groups.npy")
    label_path = args.output_path.replace(".npy", "_dim_labels.npy")

    np.save(base_path, base_codes)
    np.save(group_path, np.array(groups, dtype=object), allow_pickle=True)
    np.save(label_path, dim_labels)

    print("=" * 80)
    print(f"Saved final SID to: {args.output_path}")
    print(f"Saved base SID to: {base_path}")
    print(f"Saved dimension groups to: {group_path}")
    print(f"Saved dimension labels to: {label_path}")


if __name__ == "__main__":
    main()
