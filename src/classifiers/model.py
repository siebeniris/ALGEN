import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(Classifier, self).__init__()
        # dropout?
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.fc(x)
