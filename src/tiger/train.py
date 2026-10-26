import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
from .dataset import GenRecDataset, DEFAULT_CODEBOOK_SIZE
from .dataloader import GenRecDataLoader

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        decoder_start_token_id=config['pad_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
    )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)
    
    @property
    def n_parameters(self):
      """Calculates the number of trainable parameters in the model.

      Returns:
          str: A string containing the number of embedding parameters,
          non-embedding parameters, and total trainable parameters.
      """
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      """Forward pass of the model. Returns the output logits and the loss value.

      Args:
          batch (dict): A dictionary containing the input data for the model.

      Returns:
          outputs (ModelOutput):
              The output of the model, which includes:
              - loss (torch.Tensor)
              - logits (torch.Tensor)
      """
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
        """Generate recommendations using the model.

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            max_length (int): Maximum length of the generated sequence.
            num_beams (int): Number of beams for beam search.

        Returns:
            torch.Tensor: Generated output tensor.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=7,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )

def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    # A generated sequence is longer than the ground-truth SID: `max_length` in
    # T5.generate() pads the decode out to a fixed number of steps, so `preds`
    # carries extra trailing tokens.  The prediction is correct iff its SID
    # prefix equals the label, so compare only the first `len(label)` tokens.
    # (Comparing full-length sequences makes every comparison fail and silently
    # reports NDCG == 0 for every epoch, which disables early stopping and model
    # selection entirely.)
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      label_len = len(cur_label)
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()[:label_len]
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  # Assume only one ground truth item per example
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_item_level(model, eval_loader, topk_list, beam_size, device,
                        code_path, codebook_size=DEFAULT_CODEBOOK_SIZE):
    """
    Item-level evaluation, matching the metric the paper reports.

    Why this exists: `evaluate()` below scores a *token prefix*, which is a
    conservative proxy (measured 14-34% below the item-level metric) and can
    select a different checkpoint.  The stored canonical checkpoints were
    selected with an item-level evaluator: their final logged validation
    NDCG@20 equals the item-level validation NDCG@20 of the saved checkpoint to
    4 decimals (Beauty K=2: 0.054237 vs 0.054187; Beauty K=4: 0.054712 vs
    0.054717).  Reproducing the canonical protocol therefore requires this
    function.

    Per user: generated tokens -> subtract the per-position slot offset -> SID
    -> item id (via the SID array) -> merge beams that map to the same item by
    summing their normalized beam probability -> rank items by merged score.
    `sequences_scores` are length-normalised log-probabilities, so a softmax
    over the valid beams is applied before merging.  Users whose ground-truth
    item was not generated contribute 0.
    """
    import numpy as _np
    codes = _np.load(code_path, allow_pickle=True).astype(_np.int64)
    sid_len = codes.shape[1]
    sid_to_item = {}
    for idx, row in enumerate(codes):
        sid_to_item[tuple(int(x) for x in row)] = idx + 1
    offsets = _np.arange(sid_len) * codebook_size + 1

    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating(item-level)"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target']

            out = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=sid_len + 1,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                return_dict_in_generate=True,
                output_scores=True,
            )
            bs = input_ids.shape[0]
            seq = out.sequences[:, 1:].cpu().numpy()
            if seq.shape[1] < sid_len:
                seq = _np.concatenate(
                    [seq, _np.zeros((seq.shape[0], sid_len - seq.shape[1]), dtype=seq.dtype)],
                    axis=1)
            seq = seq[:, :sid_len]
            if hasattr(out, 'sequences_scores') and out.sequences_scores is not None:
                sc = out.sequences_scores.cpu().numpy()
            else:
                sc = _np.zeros(seq.shape[0], dtype=_np.float64)

            # (bs, beam, sid_len); this is exactly the layout dump_beam_outputs.py uses
            seq3 = seq.reshape(bs, beam_size, sid_len)
            sc2 = sc.reshape(bs, beam_size)

            ranks = _np.full(bs, -1, dtype=_np.int64)
            for i in range(bs):
                item_probs = {}
                for b in range(beam_size):
                    sids = tuple(int(seq3[i, b, pp]) - int(offsets[pp])
                                 for pp in range(sid_len))
                    if any((x < 0) or (x >= codebook_size) for x in sids):
                        continue
                    it = sid_to_item.get(sids, -1)
                    if it <= 0:
                        continue
                    item_probs.setdefault(it, []).append(b)
                if not item_probs:
                    continue
                valid = _np.zeros(beam_size, dtype=bool)
                for bs_idx in item_probs.values():
                    for b in bs_idx:
                        valid[b] = True
                v = _np.where(valid, sc2[i], -_np.inf)
                v = v - _np.max(v)
                e = _np.where(valid, _np.exp(v), 0.0)
                tot = e.sum()
                if tot <= 0:
                    continue
                e = e / tot
                merged = {it: float(e[bs_idx].sum()) for it, bs_idx in item_probs.items()}
                # The label holds the ground-truth SID's CODES, not an item id:
                # decode the full SID through the same map used for the beams.
                lab = labels[i].cpu().numpy()
                gt_sid = tuple(int(lab[pp]) - int(offsets[pp]) for pp in range(sid_len))
                gt = sid_to_item.get(gt_sid, -1)
                if gt not in merged:
                    continue
                ranks[i] = 1 + sum(1 for v2 in merged.values() if v2 > merged[gt])

            for k in topk_list:
                hit = (ranks >= 1) & (ranks <= k)
                nd = _np.zeros_like(ranks, dtype=float)
                if hit.any():
                    nd[hit] = 1.0 / _np.log2(ranks[hit] + 1)
                recalls['Recall@' + str(k)].append(float(hit.mean()))
                ndcgs['NDCG@' + str(k)].append(float(nd.mean()))

    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def evaluate(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            # print(f"pos_index shape: {pos_index.shape}, pos_index: {pos_index}")
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes (0 = main process only; use 0 on restricted/CPU-only hosts)')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--codebook_size', type=int, default=DEFAULT_CODEBOOK_SIZE,
                        help='Token slot width used by item2code '
                             '(token = code + pos*codebook_size + 1). '
                             f'Default {DEFAULT_CODEBOOK_SIZE}.')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/tiger.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    parser.add_argument('--eval_mode', type=str, default='item', choices=['item', 'token'],
                        help='Validation metric used for early stopping / checkpoint '
                             "selection. 'item' (default) decodes SIDs to item ids and is "
                             'the metric the paper reports; "token" is the legacy token-'
                             'prefix proxy, which is 14-34%% lower and can select a '
                             'different checkpoint.')
    config = vars(parser.parse_args())
    # Create directories for logs and checkpoints
    if os.path.dirname(config['log_path']):
        os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    if os.path.dirname(config['save_path']):
        os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    # Set random seed for reproducibility
    set_seed(config['seed'])

    # Initialize model
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)
    # Check if the device is available
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        codebook_size=config['codebook_size'],
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        codebook_size=config['codebook_size'],
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        codebook_size=config['codebook_size'],
    )

    # The token layout is `token = code + pos*codebook_size + 1`, so the model
    # vocabulary must be exactly sid_len*codebook_size + 1.  A mismatch means the
    # checkpoint and the SID array disagree about the slot width, which silently
    # corrupts every decoded item id -- fail before training rather than after.
    sid_len = len(next(iter(train_dataset.item_to_code.values())))
    expected_vocab = sid_len * config['codebook_size'] + 1
    if config['vocab_size'] != expected_vocab:
        raise ValueError(
            f"vocab_size={config['vocab_size']} is inconsistent with "
            f"sid_len={sid_len} and codebook_size={config['codebook_size']} "
            f"(expected {expected_vocab} = {sid_len}*{config['codebook_size']}+1). "
            f"Pass --vocab_size {expected_vocab}."
        )
    logging.info(f"SID length={sid_len}, codebook_size={config['codebook_size']}, "
                 f"vocab_size={config['vocab_size']} (consistent)")
    print(f"SID length={sid_len}, codebook_size={config['codebook_size']}, "
          f"vocab_size={config['vocab_size']}")

    nw = int(config.get('num_workers', 4))
    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=nw)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False, num_workers=nw)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False, num_workers=nw)
    
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(validation_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")
    # for batch in train_dataloader:
    #     print(f"Batch size: {len(batch['history'])}")
    #     print(f"the first batch history:{batch['history'][0]}")
    #     print(f"the first batch target:{batch['target'][0]}")
    #     print(f"the first batch attention mask:{batch['attention_mask'][0]}")
    #     break

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0
    saved_any = False
    
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")
        # Evaluate the model
        if config.get('eval_mode', 'item') == 'item':
            avg_recalls, avg_ndcgs = evaluate_item_level(
                model, validation_dataloader, config['topk_list'], config['beam_size'],
                device, config['code_path'], config['codebook_size'])
        else:
            avg_recalls, avg_ndcgs = evaluate(
                model, validation_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset: {test_avg_recalls}")
            logging.info(f"Test Dataset: {test_avg_ndcgs}")
            # Save the best model
            torch.save(model.state_dict(), config['save_path'])
            saved_any = True
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break

    if not saved_any:
        # Validation NDCG@20 stayed at 0 for every epoch (possible on tiny or
        # synthetic data).  Still save the final model so that the pipeline can
        # continue; the log records that this is not a validation-selected model.
        torch.save(model.state_dict(), config['save_path'])
        logging.warning(f"No epoch improved validation NDCG@20; saved the final "
                        f"model to {config['save_path']} (not validation-selected)")
        
