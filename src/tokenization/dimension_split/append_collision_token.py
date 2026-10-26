import argparse
import numpy as np
from collections import defaultdict


def compute_collision(codes):
    tuples = [tuple(x.tolist()) for x in codes]
    unique_num = len(set(tuples))
    total_num = len(tuples)
    return 1 - unique_num / total_num, unique_num, total_num


def append_collision_token(input_path, output_path, codebook_size=256):
    codes = np.load(input_path, allow_pickle=True).astype(np.int64)

    print("=" * 80)
    print("Input path:", input_path)
    print("Original SID shape:", codes.shape)

    old_collision, old_unique, total = compute_collision(codes)
    print("Original total items:", total)
    print("Original unique SID:", old_unique)
    print("Original collision rate:", old_collision)

    groups = defaultdict(list)
    for idx, row in enumerate(codes):
        groups[tuple(row.tolist())].append(idx)

    max_group_size = max(len(v) for v in groups.values())
    print("Max duplicate group size:", max_group_size)

    if max_group_size > codebook_size:
        raise ValueError(
            f"Maximum duplicate-group size {max_group_size} exceeds codebook_size={codebook_size}."
            f"A single disambiguation token is not enough; increase the codebook or regenerate the base SIDs."
        )

    extra = np.zeros((codes.shape[0], 1), dtype=np.int64)

    duplicate_group_num = 0
    duplicate_item_num = 0

    for sid, indices in groups.items():
        if len(indices) > 1:
            duplicate_group_num += 1
            duplicate_item_num += len(indices)

        for local_id, item_idx in enumerate(indices):
            extra[item_idx, 0] = local_id

    new_codes = np.concatenate([codes, extra], axis=1)

    new_collision, new_unique, _ = compute_collision(new_codes)

    print("=" * 80)
    print("Duplicate SID groups:", duplicate_group_num)
    print("Items in duplicate groups:", duplicate_item_num)
    print("New SID shape:", new_codes.shape)
    print("New unique SID:", new_unique)
    print("New collision rate:", new_collision)
    print("Max disambiguation token:", int(extra.max()))

    np.save(output_path, new_codes)

    print("=" * 80)
    print("Saved new SID to:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--codebook_size", type=int, default=256)
    args = parser.parse_args()

    append_collision_token(
        input_path=args.input_path,
        output_path=args.output_path,
        codebook_size=args.codebook_size,
    )
