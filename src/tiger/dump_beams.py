import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

try:  # package execution: python3 -m src.tiger.dump_beams
    from .train import TIGER, set_seed
    from .dataset import GenRecDataset
    from .dataloader import GenRecDataLoader
except ImportError:  # direct script execution: python3 src/tiger/dump_beams.py
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from src.tiger.train import TIGER, set_seed
    from src.tiger.dataset import GenRecDataset
    from src.tiger.dataloader import GenRecDataLoader

# Token layout used at training time: token = code + i*codebook_size + 1.
# The spectral/residual SIDs in this project are 256-ary per position, so the
# slot width MUST be 256. The previous default of 1024 silently mismatched the
# training-time layout (vocab_size == sid_len*256 + 1 confirms the slot width).
DEFAULT_CODEBOOK_SIZE = 256


def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def infer_vocab_size_from_state_dict(state_dict):
    for key in [
        "model.shared.weight",
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "model.lm_head.weight",
    ]:
        if key in state_dict:
            return state_dict[key].shape[0]

    for k, v in state_dict.items():
        if k.endswith("shared.weight") or k.endswith("embed_tokens.weight"):
            return v.shape[0]

    raise ValueError("Cannot infer vocab_size from checkpoint.")


def build_config(vocab_size):
    return {
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "vocab_size": vocab_size,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "feed_forward_proj": "relu",
    }


def build_sid_to_item(code_path):
    """
    Build {sid_tuple -> item_id}.

    NOTE: item ids are 1-based everywhere else in the pipeline
    (`item2code` maps row i to item i+1, and the training parquet `user`/
    `history`/`target` columns are 1-based). This function used to store the
    0-based row index, which silently shifted every decoded item id by -1
    relative to `candidate_tokens`. Keep it 1-based.
    """
    codes = np.load(code_path, allow_pickle=True).astype(np.int64)
    mp = {}
    duplicate = 0

    for idx, row in enumerate(codes):
        key = tuple(row.tolist())
        if key in mp:
            duplicate += 1
        mp[key] = idx + 1

    print("Code path:", code_path)
    print("Code shape:", codes.shape)
    print("Duplicate SID count:", duplicate)

    return mp, codes


def tokens_to_sid(tokens, sid_len, codebook_size=DEFAULT_CODEBOOK_SIZE):
    tokens = [int(x) for x in tokens[:sid_len]]

    if len(tokens) < sid_len:
        return None

    sid = []
    for i, t in enumerate(tokens):
        low = i * codebook_size + 1
        high = (i + 1) * codebook_size

        if t < low or t > high:
            return None

        sid.append(t - low)

    return tuple(sid)


def decode_tokens_to_item(tokens, sid_len, sid_to_item, codebook_size=DEFAULT_CODEBOOK_SIZE):
    sid = tokens_to_sid(tokens, sid_len, codebook_size=codebook_size)
    if sid is None:
        return -1
    return sid_to_item.get(sid, -1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="../data/Beauty")
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--sid_len", type=int, required=True)
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=DEFAULT_CODEBOOK_SIZE,
        help="Token slot width. Must match the codebook_size used when training "
             "(item2code: token = code + i*codebook_size + 1). Default 256.",
    )
    parser.add_argument("--beam_size", type=int, default=30)
    parser.add_argument("--infer_size", type=int, default=96)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--allow-low-decode", action="store_true",
                        help="EXPERTS ONLY / synthetic smoke: allow a candidate "
                             "decode ratio below 0.90 instead of raising. A low ratio "
                             "normally means the token layout (--codebook_size/--sid_len) "
                             "does not match training.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (0 = main process only)")

    args = parser.parse_args()

    set_seed(args.seed)

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = strip_module_prefix(state_dict)

    vocab_size = infer_vocab_size_from_state_dict(state_dict)
    print("Inferred vocab_size from ckpt:", vocab_size)

    config = build_config(vocab_size)
    model = TIGER(config)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sid_to_item, codes = build_sid_to_item(args.code_path)

    data_file = os.path.join(args.dataset_path, f"{args.split}.parquet")

    dataset = GenRecDataset(
        dataset_path=data_file,
        code_path=args.code_path,
        mode="evaluation",
        max_len=args.max_len,
    )

    loader = GenRecDataLoader(
        dataset,
        batch_size=args.infer_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_target_item_ids = []
    all_candidate_item_ids = []
    all_candidate_tokens = []
    all_sequence_scores = []
    all_transition_logprobs = []

    gen_max_length = args.sid_len + 1

    print("=" * 80)
    print("Dumping beam outputs")
    print("Split:", args.split)
    print("SID length:", args.sid_len)
    print("Generate max_length:", gen_max_length)
    print("Beam size:", args.beam_size)
    print("=" * 80)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Dumping {args.split}"):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)

            batch_size = input_ids.shape[0]

            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max_length,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                return_dict_in_generate=True,
                output_scores=True,
            )

            seq = outputs.sequences
            seq_wo_start = seq[:, 1:]

            if seq_wo_start.shape[1] < args.sid_len:
                pad = torch.zeros(
                    (seq_wo_start.shape[0], args.sid_len - seq_wo_start.shape[1]),
                    dtype=seq_wo_start.dtype,
                    device=seq_wo_start.device,
                )
                seq_wo_start = torch.cat([seq_wo_start, pad], dim=1)

            seq_wo_start = seq_wo_start[:, :args.sid_len]
            candidate_tokens = seq_wo_start.reshape(batch_size, args.beam_size, args.sid_len)

            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                sequence_scores = outputs.sequences_scores.reshape(batch_size, args.beam_size)
            else:
                sequence_scores = torch.zeros((batch_size, args.beam_size), device=device)

            try:
                transition_scores = model.model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    outputs.beam_indices,
                    normalize_logits=True,
                )
                transition_scores = transition_scores[:, :args.sid_len]
                if transition_scores.shape[1] < args.sid_len:
                    pad = torch.full(
                        (transition_scores.shape[0], args.sid_len - transition_scores.shape[1]),
                        float("nan"),
                        device=transition_scores.device,
                    )
                    transition_scores = torch.cat([transition_scores, pad], dim=1)

                transition_scores = transition_scores.reshape(batch_size, args.beam_size, args.sid_len)
            except Exception as e:
                print("Warning: cannot compute transition scores:", e)
                transition_scores = torch.full(
                    (batch_size, args.beam_size, args.sid_len),
                    float("nan"),
                    device=device,
                )

            labels_np = labels.detach().cpu().numpy()
            cand_tokens_np = candidate_tokens.detach().cpu().numpy()

            target_item_ids = []
            candidate_item_ids = np.full((batch_size, args.beam_size), -1, dtype=np.int64)

            for i in range(batch_size):
                target_item = decode_tokens_to_item(
                    labels_np[i], args.sid_len, sid_to_item, args.codebook_size
                )
                target_item_ids.append(target_item)

                for b in range(args.beam_size):
                    item_id = decode_tokens_to_item(
                        cand_tokens_np[i, b], args.sid_len, sid_to_item, args.codebook_size
                    )
                    candidate_item_ids[i, b] = item_id

            all_target_item_ids.append(np.asarray(target_item_ids, dtype=np.int64))
            all_candidate_item_ids.append(candidate_item_ids)
            all_candidate_tokens.append(cand_tokens_np.astype(np.int64))
            all_sequence_scores.append(sequence_scores.detach().cpu().numpy().astype(np.float32))
            all_transition_logprobs.append(transition_scores.detach().cpu().numpy().astype(np.float32))

    target_item_ids = np.concatenate(all_target_item_ids, axis=0)
    candidate_item_ids = np.concatenate(all_candidate_item_ids, axis=0)
    candidate_tokens = np.concatenate(all_candidate_tokens, axis=0)
    sequence_scores = np.concatenate(all_sequence_scores, axis=0)
    transition_logprobs = np.concatenate(all_transition_logprobs, axis=0)

    # --- decode health check -------------------------------------------------
    # If the token layout (--codebook_size) does not match the one used at
    # training time, virtually every candidate fails to decode and every id
    # becomes -1, which silently destroys all downstream metrics. Fail loudly.
    valid_ratio = float((candidate_item_ids >= 0).mean())
    target_ratio = float((target_item_ids >= 0).mean())
    print("=" * 72)
    print(f"Decode health: candidate_item_ids valid ratio = {valid_ratio:.4f}")
    print(f"Decode health: target_item_ids    valid ratio = {target_ratio:.4f}")
    print(f"codebook_size = {args.codebook_size}, sid_len = {args.sid_len}, "
          f"inferred vocab = {int(vocab_size)}")
    if valid_ratio < 0.90 and args.allow_low_decode:
        print("WARNING: candidate decode ratio {:.1%} < 90%; continuing because "
              "--allow-low-decode was passed (synthetic/untrained checkpoint)".format(valid_ratio))
    elif valid_ratio < 0.90:
        raise RuntimeError(
            "Only {:.1%} of candidates decoded to a valid item id. The token "
            "layout is almost certainly wrong: check --codebook_size "
            "(must equal the codebook_size used by item2code at training time) "
            "and --sid_len.".format(valid_ratio)
        )
    if target_ratio < 0.99:
        print(f"WARNING: only {target_ratio:.2%} of ground-truth targets decoded; "
              "the dataset code_path may not match the one used for training.")
    print("=" * 72)

    valid_ratio = (candidate_item_ids >= 0).mean()
    target_valid_ratio = (target_item_ids >= 0).mean()

    print("=" * 80)
    print("Finished dumping")
    print("target_item_ids shape:", target_item_ids.shape)
    print("candidate_item_ids shape:", candidate_item_ids.shape)
    print("candidate_tokens shape:", candidate_tokens.shape)
    print("sequence_scores shape:", sequence_scores.shape)
    print("transition_logprobs shape:", transition_logprobs.shape)
    print("Candidate valid ratio:", valid_ratio)
    print("Target valid ratio:", target_valid_ratio)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.savez_compressed(
        args.output_path,
        split=np.array(args.split),
        code_path=np.array(args.code_path),
        ckpt_path=np.array(args.ckpt_path),
        sid_len=np.array(args.sid_len),
        beam_size=np.array(args.beam_size),
        vocab_size=np.array(vocab_size),
        target_item_ids=target_item_ids,
        candidate_item_ids=candidate_item_ids,
        candidate_tokens=candidate_tokens,
        sequence_scores=sequence_scores,
        transition_logprobs=transition_logprobs,
    )

    print("Saved to:", args.output_path)


if __name__ == "__main__":
    main()
