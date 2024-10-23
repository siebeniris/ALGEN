import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers


from src.model_utils import (
    load_encoder_decoder,
    load_embedder_and_tokenizer,
    load_tokenizer
)

logger = logging.getLogger(__name__)


class FewShotInversionModel(transformers.PreTrainedModel):
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

    def __init__(self):
        super().__init__()

