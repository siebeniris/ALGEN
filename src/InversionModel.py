import torch

from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
import torch.nn as nn
from torch.nn import LayerNorm

import transformers.models.t5.modeling_t5 as t5_modeling

t5_modeling.T5LayerNorm = LayerNorm

from utils import get_device
from alignment_models import LinearAligner, NeuralAligner, EmbeddingAlignerOrthogonal
from embeddingAlingerOT import AlignerOT


###################################################################################################
### limitation of this inversionmodel: only works on the embeddings with the same tokenizers.


class EmbeddingInverter(torch.nn.Module):
    def __init__(self,
                 model_G_name_or_path: str = "t5-base",
                 model_S_name_or_path: str = "t5-small",
                 max_length: int = 128,
                 align_method="linear",
                 decoding_strategy="beam",
                 adjust_weights_with_magnitutde=True,
                 ot_reg=0.1, ot_reg_m=10.0
                 ):
        super().__init__()
        self.device = get_device()
        print(f"Model Using device: {self.device}")

        # Load tokenizers and models
        self.tokenizer_G = T5Tokenizer.from_pretrained(model_G_name_or_path)
        self.tokenizer_S = T5Tokenizer.from_pretrained(model_S_name_or_path)

        self.model_G = T5ForConditionalGeneration.from_pretrained(model_G_name_or_path)
        self.encoder_G = self.model_G.encoder
        self.encoder_S = T5Model.from_pretrained(model_S_name_or_path).encoder

        self.max_length = max_length

        self.hidden_size_G = self.encoder_G.config.hidden_size  # 768
        self.hidden_size_S = self.encoder_S.config.hidden_size  # 512,

        self.decoding_strategy = decoding_strategy

        # TODO : EXPERIEMENT ON DIFFERENT LAYERS.
        self.layer_num = 0
        self.align_method = align_method

        # Define aligner
        if self.align_method == "linear":
            print(f"initializing in the model source emb {self.hidden_size_S}, {self.hidden_size_G}")
            self.aligner = LinearAligner(self.hidden_size_S, self.hidden_size_G)
        elif self.align_method == "neural":
            self.aligner = NeuralAligner(self.hidden_size_S, self.hidden_size_G)
        elif self.align_method == "orthogonal":
            self.aligner = EmbeddingAlignerOrthogonal(self.hidden_size_S, self.hidden_size_G, orthogonal=True)
        elif self.align_method == "ot":
            self.aligner = AlignerOT(self.hidden_size_S, self.hidden_size_G, self.device)
        else:
            raise ValueError(f"Unkown Align Method: {align_method}")

        # Move models to device
        self.model_G = self.model_G.to(self.device)
        self.encoder_S = self.encoder_S.to(self.device)
        self.aligner = self.aligner.to(self.device)
        self.freeze_models()

        self.use_ff_dropout = False
        self.num_repeat_tokens = 16

        self.embedding_transform_g_g = nn.Sequential(
            nn.Linear(self.hidden_size_G, self.hidden_size_G),
            nn.Dropout(
                self.model_G.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(self.hidden_size_G, self.hidden_size_G * self.num_repeat_tokens),
        )

    def freeze_models(self):
        """Freeze model parameters to prevent gradient updates."""
        for param in self.model_G.parameters():
            param.requires_grad = False
        for param in self.encoder_S.parameters():
            param.requires_grad = False

    def get_embeddings_S(self, text):
        """Get and align embeddings from encoder S."""
        if isinstance(text, str):
            text = [text]
        # prefixed_text = ["translate English to English: " + t for t in text]

        inputs = self.tokenizer_S(
            text,
            return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            embeddings_s = self.encoder_S(**inputs).last_hidden_state
            # aligned_embeddings = self.aligner(embeddings_s)

        return embeddings_s, inputs["input_ids"], inputs["attention_mask"]

    def get_embeddings_G(self, text):
        """Get embeddings from encoder G """
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer_G(text, return_tensors="pt", padding=True, max_length=self.max_length,
                                  truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.encoder_G(**inputs).last_hidden_state
        return embeddings, inputs["input_ids"], inputs["attention_mask"]

    def decode_embeddings(self, embeddings, attention_mask=None):
        """Decode embeddings back to text."""

        if self.align_method == "ot":
            # seq_len, hidden_size = embeddings.shape
            embeddings = embeddings.unsqueeze(0)
        print("decoding embeddings shape: ", embeddings.shape)
        with torch.no_grad():
            batch_size, seq_length, hidden_size = embeddings.size()
            print(f"embeddings mean: {embeddings.mean().item()}, std: {embeddings.std().item()}")

            if attention_mask == None:
                print(f"all ones for attention_mask")
                attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=embeddings.device)

            if embeddings.size(0) == 0 or attention_mask.size(0) == 0:
                print("Error: Empty embeddings or attention mask.")
                return ["Error during generation: Empty input"] * batch_size

            # Create encoder outputs
            encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)
            print(f"Encoder outputs last_hidden_state shape: {encoder_outputs.last_hidden_state.shape}")

            # Initialize decoder input IDs
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer_G.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            assert embeddings.size(0) == decoder_input_ids.size(0), "Decoder input batch size mismatch!"

            # print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
            try:
                if self.decoding_strategy == "beam":
                    # Generate output
                    generated = self.model_G.generate(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.max_length,
                        num_beams=5,
                        length_penalty=1.0,
                        early_stopping=True,
                        pad_token_id=self.tokenizer_G.pad_token_id,
                        eos_token_id=self.tokenizer_G.eos_token_id,
                    )
                elif self.decoding_strategy == "neucleus":
                    generated = self.model_G.generate(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.max_length,
                        do_sample=True,
                        top_p=0.9,
                        # dynamically selects tokens based on cumulative probability, making it more adaptive to different probability distributions
                        temperature=0.7,  # set temperature <1, output more confident and reduce randomness.
                        pad_token_id=self.tokenizer_G.pad_token_id,
                        eos_token_id=self.tokenizer_G.eos_token_id,
                    )

                # Decode generated tokens
                # print(f"Generated IDs: {generated}")
                # print(f"Generated shape: {generated.shape}")

                decoded_text = self.tokenizer_G.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                decoded_text = [text.strip() for text in decoded_text]

                print(f"decoded text:", decoded_text)

            except Exception as e:
                print(f"Error during generation: {str(e)}")
                decoded_text = ["Error during generation"] * batch_size
        return decoded_text

    def transform_and_concatenate(self, embedding_g, aligned_s, attention_mask_s):
        batch_size, seq_len, hidden_dim_g = embedding_g.shape
        # (b,s,g)
        diff_embedding = embedding_g - aligned_s
        diff_embedding = diff_embedding.reshape((batch_size, self.num_repeat_tokens,
                                                 self.hidden_size_G))
        # (b,s,g)
        embedding_g = self.embedding_transform_g_g(embedding_g)
        embedding_g = embedding_g.reshape((batch_size, self.num_repeat_tokens,
                                           self.hidden_size_G))

        aligned_s = self.embedding_transform_g_g(aligned_s)
        aligned_s = aligned_s.reshape((batch_size, self.num_repeat_tokens,
                                       self.hidden_size_G))

        ones = torch.ones(
            (batch_size, 1), dtype=torch.long, device=aligned_s.device
        )
        sep_token = ones * self.model_G.config.eos_token_id
        sep_token = self.encoder_G.embed_tokens(sep_token)

        concat_embeds = torch.cat(
            (
                sep_token,
                embedding_g,
                sep_token,
                aligned_s,
                sep_token,
                diff_embedding
            ),
            dim=1
        )
        attention_mask = torch.cat((ones.repeat(1, 3 + 3 * self.num_repeat_tokens), attention_mask_s),
                                   dim=1, )
        return concat_embeds, attention_mask

    def forward(self, x):
        """Forward pass for text-to-text inversion."""
        # aligning: linear and neural.
        # text [batch_size, seq_len]
        embedding_s = x["emb_s"]
        s_attention_mask = x["attention_mask_s"]

        embedding_g = x["emb_g"]
        g_attention_mask = x["attention_mask_g"]
        # print(f"emb_s {embedding_s.shape}, emb_g {embedding_g.shape}")

        assert embedding_s.shape[1] == embedding_g.shape[1]  # assert they have the same tokenizer

        if self.align_method == "linear":
            aligned_embeddings = self.aligner(embedding_s)

        elif self.align_method == "neural":
            aligned_embeddings = self.aligner(embedding_s)

        elif self.align_method == "orthogonal":
            aligned_embeddings = self.aligner(embedding_s)

        elif self.align_method == "ot":
            aligned_embeddings = self.aligner(embedding_s[0], embedding_g[0])

        else:
            raise ValueError(f"Unkown Align Method: {self.align_method}")

        # # TODO: HOW TO DEAL WITH THIS
        # input_embeds , attention_mask = self.transform_and_concatenate(embedding_g, aligned_embeddings, s_attention_mask)

        # print(aligned_embeddings.shape, attention_mask.shape, aligned_embeddings.device)
        # we can use attention_mask from embedding_s only when the seq_len is not changed for embedding_s
        # only when they have the same kind of tokenizer.
        # return aligned_embeddings, self.decode_embeddings(aligned_embeddings, s_attention_mask)
        return aligned_embeddings

    def sanity_check_random_embedding(self):
        """Check if T5 can decode random embeddings."""
        # Set up random embeddings
        hidden_size = self.model_G.config.hidden_size
        batch_size = 2
        seq_length = 10

        random_embeddings = torch.randn(batch_size, seq_length, hidden_size).to(self.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=random_embeddings)

        # Set up attention mask
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long).to(self.device)

        # Set up decoder input IDs
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer_G.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        print("embedding shape:", random_embeddings.shape)
        print("attention mask shape:", random_embeddings.shape)

        try:
            # Run the generate method
            generated = self.model_G.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=20,
                num_beams=2,
                early_stopping=True
            )

            # Decode generated tokens
            decoded_text = self.tokenizer_G.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(f"Generated Text from Random Embeddings: {decoded_text}")

        except Exception as e:
            print(f"Error during generation: {str(e)}")

    def sanity_check(self):
        # Simple decoder input for generation
        simple_input_ids = torch.tensor([[self.tokenizer_G.pad_token_id]], device=self.device)

        # Test basic generation
        try:
            generated_simple = self.model_G.generate(
                input_ids=simple_input_ids,
                max_length=10,
                num_beams=2,
                early_stopping=True
            )
            print(f"Generated IDs (basic test): {generated_simple}")
        except Exception as e:
            print(f"Error during basic generation: {str(e)}")
