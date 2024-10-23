import abc
import functools
import hashlib
import json
import os
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import pairwise
import evaluate
import transformers
from transformers import default_data_collator


from metrics_utils import CosineSimilarityLoss, evaluate_bleu
from model_utils import load_embedder_and_tokenizer, load_tokenizer, load_encoder_decoder


class Experiment(abc.ABC):
    def __init__(self,
                 model_args,
                 data_args,
                 training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def load_tokenizer(selfs) -> transformers.PreTrainedTokenizer:


encoder_model_name = "intfloat/multilingual-e5-small"
encoder_decoder_model_name = "google-t5/t5-small"
encoder, encoder_tokenizer = load_embedder_and_tokenizer("me5")
encoder_decoder = load_encoder_decoder(encoder_decoder_model_name)
encoder_decoder_tokenizer = load_tokenizer(encoder_decoder_model_name, max_length=128)
