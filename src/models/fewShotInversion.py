import copy
import logging
from typing import Dict, Optional, Tuple

import ot
import torch
import torch.nn as nn
import transformers
from config import InversionConfig

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
    embedding_aligner: nn.Module # module that align the embeddings from embedder to encoder_decoder embeddings.

    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool
    embedder_model_api: Optional[str]

    def __init__(self, config:InversionConfig):
        super().__init__(config=config)
        # get from model/dataset/training configs.
        self.embedder_model_api = config.embedder_model_api
        self.encoder_decoder = load_encoder_decoder(
            model_name=config.encoder_decoder_name_or_path,
            lora=config.use_lora
        )
        self.embedder, self.embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name
        )
        self.tokenizer = load_tokenizer(
            name=config.encoder_decoder_name_or_path,
            max_length=config.max_seq_length
        )

        # encoder_decoder's encoder hidden size.
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size

        if self.embedder_model_api:
            self.embedder_dim = config.embedder_dim
        else:
            self.embedder_dim = self.embedder.config.hidden_size

        self.alignment_strategy = config.aligning_strategy




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
        embeddings = mean_pool(hidden_state, attention_mask )
        return embeddings


    def _procrustes_algin(self, X, Y):
        # align X to Y's embedding space.
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
        C = ot.dist(X,Y, metric="sqeuclidean")
        # optimal transport plan with entropic regularization
        n, m = X.shape[0], Y.shape[0]
        # shape (n,m)
        T = ot.sinkhorn(torch.ones(n) / n, torch.ones(m) / m, C, reg)  # Shape (n, m)

        # align X
        X_aligned = T @ Y
        return X_aligned


    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask:torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        embedder = self.embedder
        embedder.eval()
        model_output = embedder(input_ids=input_ids, attention_mask=attention_mask)
        embedder_embeddings = self._process_embeddings(model_output, attention_mask)

        encoder = self.encoder_decoder.encoder
        encoder_output = encoder(input_ids= input_ids, attention_mask=attention_mask)
        encoder_embeddings = self._process_embeddings(encoder_output, attention_mask)

        # TODO:implement alignment here.
        if self.embedder_dim != self.encoder_hidden_dim:
            self.linear_aligner = nn.Linear(self.embedd_dim, self.encoder_hidden_dim)
            embedder_embeddings = self.alginer(embedder_embeddings)

        # TODO: do we need a transform?
        # self.embedding_transform = nn.Sequential(
        #     nn.Linear(self.embedder_dim, bottleneck_dim),
        #     nn.Dropout(self.encoder_decoder.config.dropout_rate),
        #     nn.GELU(),  # TODO consider dropout or normalization here.
        #     nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        # )

        if self.alignment_strategy == "svd":
            embedder_embeddings = self._procrustes_algin(embedder_embeddings, encoder_embeddings)
        elif self.alignment_strategy == "ot":
            embedder_embeddings = self._optimal_transport_align(embedder_embeddings, encoder_embeddings)

        attention_mask = torch.ones(
            (embedder_embeddings.shape[0], embedder_embeddings.shape[1]), device=embedder_embeddings.device
        )

        return embedder_embeddings, attention_mask










