import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data.
    """
    # Load parquet data
    data = pd.read_parquet(file_path)

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    # Apply padding or truncation
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
DEFAULT_CODEBOOK_SIZE = 256
"""Canonical codebook (slot) width for every SID array in this repository.

The training-time token layout is::

    token = code + position * codebook_size + 1

so a codebook-size mismatch does not merely shift ids -- it produces
out-of-range tokens and silently turns every decoded candidate into ``-1``.
The default used to be 1024 here while training used 256, which left the
encode/decode path inconsistent between callers.  ``DEFAULT_CODEBOOK_SIZE`` is
the single source of truth; pass ``codebook_size`` explicitly (CLI/YAML)
whenever a different width is genuinely in use.
"""


def infer_codebook_size_from_vocab(vocab_size, sid_len):
    """Recover the slot width from a model vocabulary size.

    ``vocab_size == sid_len * codebook_size + 1`` (the ``+1`` accounts for the
    padding token 0).  This is the only *authoritative* way to recover the
    training-time layout: the numeric range of a SID array alone cannot
    distinguish a codebook of ``2**ceil(log2(max_code+1))`` from a larger one.
    """
    if sid_len <= 0:
        raise ValueError(f"sid_len must be positive, got {sid_len}")
    rem = int(vocab_size) - 1
    if rem % sid_len != 0:
        raise ValueError(
            f"vocab_size={vocab_size} is not of the form "
            f"sid_len*codebook_size+1 with sid_len={sid_len}"
        )
    return rem // sid_len


def item2code(code_path, codebook_size=DEFAULT_CODEBOOK_SIZE):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :param codebook_size: token slot width used at *training* time. Defaults to
        ``DEFAULT_CODEBOOK_SIZE`` (256). Must match the value the checkpoint was
        trained with; see ``infer_codebook_size_from_vocab``.
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)

    if data.ndim != 2:
        raise ValueError(f"expected a 2-D SID array, got shape {data.shape}")
    if codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {codebook_size}")

    # A code outside [0, codebook_size) would be encoded into a neighbouring
    # position's slot and silently decode to a wrong item.  Fail loudly instead.
    max_code = int(np.max(data))
    if max_code >= codebook_size:
        raise ValueError(
            f"codebook_size={codebook_size} is too small for {code_path}: the "
            f"SID array contains code {max_code}. The token layout would be "
            f"ambiguous. Use the codebook_size the checkpoint was trained with."
        )

    item_to_code = {}
    code_to_item = {}
    
    # for index, code in enumerate(data):
    #     item_to_code[index + 1] = code
    #     code_to_item[tuple(code)] = index + 1
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i, c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item


class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, PAD_TOKEN=0,
                 codebook_size=DEFAULT_CODEBOOK_SIZE):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            codebook_size (int, optional): Token slot width used at training
                time. Defaults to DEFAULT_CODEBOOK_SIZE (256).
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.codebook_size = codebook_size
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path, codebook_size)
        # Process the dataset
        self.data = self._prepare_data()

    @property
    def vocab_size(self):
        """Tokens the model must cover: sid_len * codebook_size + 1."""
        sid_len = len(next(iter(self.item_to_code.values())))
        return sid_len * self.codebook_size + 1
        
    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )
        # Convert items to codes
        for item in processed_data:
            code_len = len(next(iter(self.item_to_code.values())))

            item['history'] = [
                self.item_to_code.get(x, np.array([self.PAD_TOKEN] * code_len))
                for x in item['history']
            ]

            item['target'] = self.item_to_code.get(
                item['target'],
                np.array([self.PAD_TOKEN] * code_len)
            )
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
    
if __name__ == "__main__":
    # Example usage
    dataset_path = '../data/Beauty/train.parquet'
    code_path = '../data/Beauty/Beauty_t5_rqvae.npy'
    mode = 'train'  # or 'train'
    max_len = 20

    dataset = GenRecDataset(dataset_path, code_path, mode, max_len)
    print("Number of items in dataset:", len(dataset))

    print("First five items in dataset:", [dataset[i] for i in range(5)])
    
