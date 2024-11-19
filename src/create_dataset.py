import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from InversionModel import EmbeddingInverter
from torch.nn.utils.rnn import pad_sequence

from data_helper import load_data


class EmbeddingDataset(Dataset):
    def __init__(self, texts: List[str], inverter: nn.Module):
        """
        Dataset for training embedding alignment

        Args:
            texts: List of input texts
            inverter: Guide model tokenizer

        """

        self.texts = texts
        self.inverter = inverter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        embeddings_g, input_ids_g, attention_mask_g = self.inverter.get_embeddings_G(text)
        embeddings_s, input_ids_s, attention_mask_s = self.inverter.get_embeddings_S(text)

        return {
            "text": text,
            "emb_g": embeddings_g.squeeze(0),  # [seq_len, hidden_dim]
            "emb_s": embeddings_s.squeeze(0),  # [seq_len, hidden_dim]

            "input_ids_g": input_ids_g.squeeze(0),
            "input_ids_s": input_ids_s.squeeze(0),

            "attention_mask_g": attention_mask_g.squeeze(0),
            "attention_mask_s": attention_mask_s.squeeze(0),

            "labels": input_ids_g.squeeze(0)  # this is supposed to be the eventually label.
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for the EmbeddingDataset that handles padding

    Args:
        batch: List of dictionaries containing:
            - text: str
            - emb_g: tensor [seq_len, hidden_dim]
            - emb_s: tensor [seq_len, hidden_dim]
            - input_ids_g: tensor [seq_len]
            - input_ids_s: tensor [seq_len]
            - attention_mask_g: tensor [seq_len]
            - attention_mask_s: tensor [seq_len]
            - labels: tensor [seq_len]

    Returns:
        Batched dictionary with padded tensors
    """

    # Get texts (no padding needed)
    # print("padding ....")
    texts = [item['text'] for item in batch]

    # Pad sequences for embeddings (they should have same sequence length for g and s)
    emb_g = pad_sequence([item['emb_g'] for item in batch], batch_first=True)
    emb_s = pad_sequence([item['emb_s'] for item in batch], batch_first=True)

    # Pad sequences for input_ids and attention masks
    input_ids_g = pad_sequence([item['input_ids_g'] for item in batch],
                               batch_first=True,
                               padding_value=0)  # Usually 0 is the pad token
    input_ids_s = pad_sequence([item['input_ids_s'] for item in batch],
                               batch_first=True,
                               padding_value=0)

    # Pad attention masks (with 0)
    attention_mask_g = pad_sequence([item['attention_mask_g'] for item in batch],
                                    batch_first=True,
                                    padding_value=0)
    attention_mask_s = pad_sequence([item['attention_mask_s'] for item in batch],
                                    batch_first=True,
                                    padding_value=0)

    # Pad labels (usually with -100 for T5 models)
    labels = pad_sequence([item['labels'] for item in batch],
                          batch_first=True,
                          padding_value=-100)  # -100 is typically ignored in loss calculation

    # Create lengths tensor to keep track of original sequence lengths
    lengths = torch.tensor([len(item['emb_g']) for item in batch])

    return {
        'text': texts,
        'emb_g': emb_g,
        'emb_s': emb_s,
        'input_ids_g': input_ids_g,
        'input_ids_s': input_ids_s,
        'attention_mask_g': attention_mask_g,
        'attention_mask_s': attention_mask_s,
        'labels': labels,
        'lengths': lengths  # Optional, but useful for some operations
    }


# Use it in your DataLoader:

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    text_list = load_data("flores", "eng_Latn", nr_samples=10)
    inverter = EmbeddingInverter()

    print("loading dataset")
    embedding_dataset = EmbeddingDataset(text_list, inverter)

    train_dataloader = DataLoader(
        embedding_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=10,  # if you're using multiple workers
        pin_memory=torch.cuda.is_available(),  # if you're using GPU
        persistent_workers=True if 10 > 0 else False,
        drop_last=False,
    )

    print(embedding_dataset.__getitem__(1))
    batch = next(iter(train_dataloader))
    print(batch)
