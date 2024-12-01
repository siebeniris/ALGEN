import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from InversionModel import EmbeddingInverter
from torch.nn.utils.rnn import pad_sequence

from data_helper import load_data


class EmbeddingDataset(Dataset):
    def __init__(self, X, Y, Y_attention_mask, Y_gold_text):
        """
        Dataset for training embedding alignment

        Args:
            texts: List of input texts
            inverter: Guide model tokenizer

        """

        self.X = X
        self.Y = Y
        self.Y_attention_mask = Y_attention_mask
        self.Y_gold_text = Y_gold_text

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        text = self.Y_gold_text[idx]
        X = self.X[idx]
        Y = self.Y[idx]
        Y_attention_mask = self.Y_attention_mask[idx]

        return {
            "X": X,
            "Y": Y,
            "Y_attention_mask": Y_attention_mask,
            "text": text
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for the EmbeddingDataset that handles padding

    Args:
        batch: List of dictionaries containing:
            - text: str
    Returns:
        Batched dictionary with padded tensors
    """
    texts = [item['text'] for item in batch]

    # Pad sequences for embeddings (they should have same sequence length for g and s)
    X = pad_sequence([item['X'] for item in batch], batch_first=True)
    Y = pad_sequence([item['Y'] for item in batch], batch_first=True)

    # Pad attention masks (with 0)
    X_mask = pad_sequence([item['Y_attention_mask'] for item in batch],
                                    batch_first=True,
                                    padding_value=0)
    Y_mask = pad_sequence([item['Y_attention_mask'] for item in batch],
                                    batch_first=True,
                                    padding_value=0)

    # Create lengths tensor to keep track of original sequence lengths
    lengths = torch.tensor([len(item['X']) for item in batch])

    return {
        'text': texts,
        'X': X,
        'Y': Y,
        'X_attention_mask': X_mask,
        'Y_attention_mask': Y_mask,
        'lengths': lengths  # Optional, but useful for some operations
    }

