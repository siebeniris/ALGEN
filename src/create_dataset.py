import torch
from torch.utils.data import Dataset


class TextEmbeddingDataset(Dataset):
    def __init__(self, texts, emb_g, emb_s):
        """
        :param texts: List of original input texts.
        :param emb_g: Embeddings from encoder g.
        :param emb_s: Embeddings from encoder s.
        """
        self.texts = texts
        self.emb_g = emb_g
        self.emb_s = emb_s

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'emb_g': torch.tensor(self.emb_g[idx], dtype=torch.float32),
            'emb_s': torch.tensor(self.emb_s[idx], dtype=torch.float32),
        }


