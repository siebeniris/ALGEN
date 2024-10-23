import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import pairwise
import evaluate
import transformers


class LinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearAligner, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, emb_s):
        return self.linear(emb_s)


