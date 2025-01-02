from typing import Dict

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import transformers.models.t5.modeling_t5 as t5_modeling

from inversion_utils import mean_pool, load_encoder_decoder_and_tokenizer
from utils import get_device

t5_modeling.T5LayerNorm = LayerNorm


class DecoderFinetuneModel(nn.Module):
    def __init__(self, model_name, max_length=32):
        super(DecoderFinetuneModel, self).__init__()
        self.device = get_device()
        self.encoder_decoder, self.tokenizer = load_encoder_decoder_and_tokenizer(model_name, self.device)
        self.embedder_dim = self.encoder_decoder.config.hidden_size
        bottleneck_dim = self.embedder_dim
        encoder_hidden_dim = self.embedder_dim
        self.num_repeat_tokens = max_length
        self.encoder_decoder.config.max_length = max_length

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * self.num_repeat_tokens)
        )

    def get_embeddings(self, hidden_states, attention_mask):
        # get the mean_pooled.
        embeddings = mean_pool(hidden_states, attention_mask)

        repeated_embeddings = self.embedding_transform(embeddings)
        embeddings = repeated_embeddings.reshape(
            (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
        )
        # batch_size, seq_length, hidden_dim
        # print("embeddings shape:", embeddings.shape)
        return embeddings, attention_mask

    def generate(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeds, attention_mask = self.get_embeddings(
            inputs["hidden_states"],
            inputs["attention_mask"]
        )
        batch_size, seq_length, hidden_size = input_embeds.size()
        # print("creating decoder_input_ids:")
        # decoder_input_ids are necessary because no labels are available
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id,
            # start decoding with the EOS,
            dtype=torch.long,
            device=self.device
        )

        return self.encoder_decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_length=self.num_repeat_tokens + 10,
            num_beams=3,
            repetition_penalty=2.0,
            length_penalty=2.0,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeds, attention_mask = self.get_embeddings(
            inputs["hidden_states"],
            inputs["attention_mask"]
        )
        batch_size, seq_length, hidden_size = input_embeds.size()

        assert batch_size == inputs["labels"].size(0), \
            f"Batch size mismatch: inputs_embeds={batch_size}, labels={inputs['labels'].size(0)}"

        # decoder input ids are right shifted using labels from T5 models
        output = self.encoder_decoder(inputs_embeds=input_embeds,
                                      attention_mask=attention_mask,
                                      labels=inputs["labels"],
                                      )
        # loss (1 item), logits: (batch_size, seq_length, vocab_size)
        return output
