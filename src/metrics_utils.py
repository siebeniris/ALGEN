import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import pairwise
import evaluate
import transformers
from transformers import default_data_collator


class CosineSimilarityLoss(nn.Module):
    def forward(self, aligned_emb_s, emb_g):
        cos_sim = nn.functional.cosine_similarity(aligned_emb_s, emb_g, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss


def evaluate_bleu(predictions, references):
    bleu_metric = evaluate.load('sacrebleu')
    return bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])


