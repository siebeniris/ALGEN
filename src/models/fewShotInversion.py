import copy
import logging
from typing import Dict, Optional, Tuple

import ot
import torch
import torch.nn as nn
import transformers
from src.models.config import InversionConfig

from src.model_utils import (
    load_encoder_decoder,
    load_embedder_and_tokenizer,
    load_tokenizer,
    mean_pool
)

logger = logging.getLogger(__name__)


class FewShotInversionModel(transformers.PreTrainedModel):
    config = InversionConfig
    # black-box embedder
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool
    tokenizer: transformers.PreTrainedTokenizer

    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool
    embedder_model_api: Optional[str]

    def __init__(self, config:InversionConfig):
        super().__init__(config=config)
        # get from model/dataset/training configs.
        self.embedder_model_api = config.embedder_model_api

        # load the white-box encoder and decoder.
        self.encoder_decoder = load_encoder_decoder(
            model_name=config.encoder_decoder_name_or_path,
            lora=config.use_lora
        )

        # load the white-box tokenizer
        self.tokenizer = load_tokenizer(
            name=config.encoder_decoder_name_or_path,
            max_length=config.max_seq_length
        )

        # load the black-box embedder and tokenizer
        self.embedder, self.embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name
        )

        # encoder_decoder's encoder hidden size.
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size

        if self.embedder_model_api:
            self.embedder_dim = config.embedder_dim
        else:
            self.embedder_dim = self.embedder.config.hidden_size

        self.embedder_no_grad = config.embedder_no_grad

        # TODO: should we have grad?
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

        # TODO: can put embedding transform here, sequential model with linear layers. and dropout
        self.alignment_strategy = config.aligning_strategy
        self.linear_aligner = nn.Linear(self.embedder_dim, self.encoder_hidden_dim)

        # transform from pooled embeddings back to 3D embeddings
        bottleneck_dim = self.encoder_hidden_dim
        self.num_repeat_tokens = config.num_repeat_tokens

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * self.num_repeat_tokens),
        )



    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embeddings(
            self,
            outputs: transformers.modeling_outputs.BaseModelOutput,
            attention_mask: torch.Tensor
        ) -> torch.Tensor:
        # process embedidng from both embedder and encoder_decoder's encoder
        # TODO: discuss this. (Details)
        hidden_state = outputs.last_hidden_state
        embeddings = mean_pool(hidden_state, attention_mask)
        return embeddings

    def _procrustes_align(self, X, Y):
        # align X to Y's embedding space. (black-box to white-box embedding space)
        # https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
        # the cross-covariance matrix
        # TODO: add iteration?
        C = X.T @ Y
        # SVD of C
        U, _, V_t = torch.linalg.svd(C, full_matrices=True)
        # compute the optimal rotation matrix W
        W = U @ V_t
        # transformation
        X_aligned = X @ W
        return X_aligned

    def _optimal_transport_align(self, X, Y, reg=0.01):
        # cost matrix, squared euclidean distance
        C = ot.dist(X, Y, metric="sqeuclidean")
        # optimal transport plan with entropic regularization
        n, m = X.shape[0], Y.shape[0]
        # shape (n,m)
        T = ot.sinkhorn(torch.ones(n) / n, torch.ones(m) / m, C, reg)  # Shape (n, m)

        # align X
        X_aligned = T @ Y
        return X_aligned

    def align_embeddings(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_masks: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_attention_masks: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # the same input text and attention mask to black-box and white-box encoders.
        embedder = self.embedder
        embedder.eval()
        model_output = embedder(input_ids=embedder_input_ids, attention_mask=embedder_attention_masks)
        embedder_embeddings = self._process_embeddings(model_output, embedder_attention_masks)

        encoder = self.encoder_decoder.encoder
        encoder_output = encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_masks)
        encoder_embeddings = self._process_embeddings(encoder_output, encoder_attention_masks)

        # TODO:implement alignment here.
        # 1. align the black-box embedding dimension to the white-box embedding dimension.
        if self.embedder_dim != self.encoder_hidden_dim:
            embedder_embeddings = self.linear_aligner(embedder_embeddings)

        # 2. Align embeddings using different strategies.
        if self.alignment_strategy == "svd":
            embedder_embeddings = self._procrustes_align(embedder_embeddings, encoder_embeddings)
        elif self.alignment_strategy == "ot":
            embedder_embeddings = self._optimal_transport_align(embedder_embeddings, encoder_embeddings)
        else:
            # without strategy, directly after dimension alignment.
            embedder_embeddings = embedder_embeddings

        # use repeat strategy to transform embeddings and reshape it in 3D
        repeated_embeddings = self.embedding_transform(embedder_embeddings)
        embeddings = repeated_embeddings.reshape(
            (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
        )

        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    # TODO: edit this.
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # why make a copy?
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask = self.align_embeddings(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_masks=inputs.get("embedder_attention_mask"),
            encoder_input_ids=inputs.get("encoder_input_ids"),
            encoder_attention_masks=inputs.get("encoder_attention_mask")
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
            embedder_input_ids: torch.Tensor,
            embedder_attention_masks: torch.Tensor,
            encoder_input_ids: torch.Tensor,
            encoder_attention_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask
        inputs_embeds, attention_mask = self.align_embeddings(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_masks=embedder_attention_masks,
            encoder_input_ids=encoder_input_ids,
            encoder_attention_masks=encoder_attention_masks
            )

        # labels: input_ids,
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )






