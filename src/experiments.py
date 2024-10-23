import abc
import functools
import hashlib
import json
import os
import logging
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import pairwise
import evaluate
import transformers
from transformers import default_data_collator


from run_args import ModelArguments, DataArguments, TrainingArguments
from metrics_utils import CosineSimilarityLoss, evaluate_bleu
from model_utils import load_embedder_and_tokenizer, load_tokenizer, load_encoder_decoder
from create_dataset import TextEmbeddingDataset


class Experiment(abc.ABC):
    def __init__(self,
                 model_args: ModelArguments,
                 data_args: DataArguments,
                 training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args


    def load_encoder_decoder(self) -> transformers.PreTrainedTokenizer:

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if self.model_args.use_lora:
            model_kwargs.update(
                {
                    "load_in_8bit": True,
                    "device_map": "auto",
                }
            )
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.encoder_decoder_name_or_path, **model_kwargs
        )

    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.encoder_decoder_name_or_path,
            padding="max_length",
            truncation="max_length",
            max_length= self.model_args.max_seq_length

        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Disable super annoying warning:
        # https://github.com/huggingface/transformers/issues/22638
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer

    def load_train_dataset(self):
        """
        Load or create train dataset.
        :return:
        """
        # mock data for now
        # later to put in huggingface dataset cards.
        texts = [
            "This is a test sentence.",
            "I love natural language processing.",
            "Transformers are amazing models!"
        ]











encoder_model_name = "intfloat/multilingual-e5-small"
encoder_decoder_model_name = "google-t5/t5-small"
encoder, encoder_tokenizer = load_embedder_and_tokenizer("me5")
encoder_decoder = load_encoder_decoder(encoder_decoder_model_name)
encoder_decoder_tokenizer = load_tokenizer(encoder_decoder_model_name, max_length=128)
