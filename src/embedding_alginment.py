import multiprocessing

import datasets
import torch
import transformers
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from model_utils import load_embedder_and_tokenizer,load_tokenizer, load_encoder_decoder

class InversionModel(torch.nn.Module):

    def __init__(self, encoder_s, encoder_s_tokenizer, encoder_decoder_g, encoder_decoder_g_tokenizer):
        super().__init__()
        self.encoder_s = encoder_s
        self.encoder_g = encoder_decoder_g.encoder
        self.encoder_s_tokenizer = encoder_s_tokenizer
        self.encoder_decoder_g_tokenizer = encoder_decoder_g_tokenizer

        self.alginer = torch.nn.Linear(encoder_s.config.hidden_size,
                                       encoder_decoder_g.config.hidden_size)

    def aligner(self, input_ids):
        # [samples, seq_lengths, emb_dim]
        embedding_s = self.encoder_s(input_ids).last_hidden_state
        embedding_g = self.encoder_g(input_ids).last_hidden_state

        algined= self.aligner(embedding_s)




if __name__ == '__main__':
    encoder_model_name = "intfloat/multilingual-e5-base"
    encoder_decoder_model_name = "google-t5/t5-base"
    encoder, encoder_tokenizer = load_embedder_and_tokenizer("me5")
    encoder_decoder = load_encoder_decoder(encoder_decoder_model_name)
    encoder_decoder_tokenizer = load_tokenizer(encoder_decoder_model_name, max_length=128)

