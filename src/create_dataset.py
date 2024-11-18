
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from InversionModel import EmbeddingInverter


from data_helper import load_data
class EmbeddingDataset(Dataset):
    def __init__(self, texts: List[str], inverter:  nn.Module):
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

        embeddings_g, input_ids_g, attention_mask_g = self.inverter.get_embeddings_S( text)
        embeddings_s, input_ids_s, attention_mask_s = self.inverter.get_embeddings_G(text)

        return {
            "text": text,
            "emb_g": embeddings_g.squeeze(0),  # [seq_len, hidden_dim]
            "emb_s": embeddings_s.squeeze(0),  # [seq_len, hidden_dim]
            "input_ids_g": input_ids_g.squeeze(0),
            "input_ids_s": input_ids_s.squeeze(0),
            "attention_mask_g": attention_mask_g.squeeze(0),
            "attention_mask_s": attention_mask_g.squeeze(0),
            "labels": input_ids_g.squeeze(0)  # this is supposed to be the eventually label.
        }


if __name__ == '__main__':
    text_list = load_data("flores", "eng_Latn", nr_samples=500)
    inverter = EmbeddingInverter()

    print("loading dataset")
    embedding_dataset = EmbeddingDataset(text_list, inverter)
    print(embedding_dataset.__getitem__(1))
