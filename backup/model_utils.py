import os
from typing import Any, Dict

import torch
import torch.nn as nn
import transformers


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def get_device():
    """
    Function that checks for GPU availability and returns the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def load_encoder_decoder(
        model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    # generic model that will be instantiated as one of the model classes of the library.
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer


def load_embedder_and_tokenizer(name: str, output_hidden_states: bool = False):
    model_kwargs = {
        "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": output_hidden_states,  # True output hidden states, for embedding last and first .
    }
    # TODO: check the configurations for commercial models

    if name == "me5":
        model = transformers.AutoModel.from_pretrained("intfloat/multilingual-e5-base", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = transformers.AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    return model, tokenizer


def get_encoder_embeddings(model: transformers, tokenizer, input_texts):
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in input_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            # consider others.
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # Use mean pooling
            embeddings.append(embedding)
            print(embedding.shape)

    return torch.tensor(embeddings)
