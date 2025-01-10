import torch
from transformers.modeling_outputs import BaseModelOutput
from torch.nn import LayerNorm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import numpy as np
import random

import transformers.models.t5.modeling_t5 as t5_modeling

t5_modeling.T5LayerNorm = LayerNorm

device = "cuda" if torch.cuda.is_available() else "cpu"
# set up rouge scores

cos = torch.nn.CosineSimilarity(dim=1)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_embeddings(embeddings,
                      attention_mask,
                      model_G, tokenizer_G,
                      num_beams=3,
                      do_sample=False,
                      repetition_penalty=2.0,
                      length_penalty=2.0,
                      top_k=None, top_p=None, temperature=None,
                      decoder_input_ids=None, max_length=32):
    """Decode embeddings back to text."""
    with torch.no_grad():
        embeddings = embeddings.to(torch.float32)
        batch_size, seq_length, hidden_size = embeddings.size()
        print(f"embeddings mean: {embeddings.mean().item()}, std: {embeddings.std().item()}")

        if attention_mask == None:
            print(f"all ones for attention_mask")
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=embeddings.device)

        if embeddings.size(0) == 0 or attention_mask.size(0) == 0:
            print("Error: Empty embeddings or attention mask.")
            return ["Error during generation: Empty input"] * batch_size

        # Create encoder outputs
        # Wrap embeddings in BaseModelOutput to simulate encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)
        # print(f"Encoder outputs last_hidden_state shape: {encoder_outputs.last_hidden_state.shape}")
        # Initialize decoder input IDs
        if decoder_input_ids == None:
            print("creating decoder_input_ids:")
            decoder_input_ids = torch.full(
                (batch_size, 1),
                1,  # tokenizer_G.eos_token_id, # start decoding with the EOS,
                dtype=torch.long,
                device=device
            )
        print("decoder input_ids:", decoder_input_ids.shape)
        assert embeddings.size(0) == decoder_input_ids.size(0), "Decoder input batch size mismatch!"

        try:
            # Generate output
            generated = model_G.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # initialized with [1], eos for the whole batch.
                max_length=max_length + 10,
                num_beams=num_beams,  # Increased beam size
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stopping=True,
                pad_token_id=tokenizer_G.pad_token_id,
                eos_token_id=tokenizer_G.eos_token_id,
            )
            decoded_text = tokenizer_G.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            decoded_text = [text.strip() for text in decoded_text]
            # print(f"decoded text:", decoded_text)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            decoded_text = ["Error during generation"] * batch_size
    return decoded_text


def add_punctuation_token_ids(sentence, tokenizer, max_length, device, punctuations=[".", "?", "!"]):
    # add punctuation to tokens ids when there is None, improve the decoding results.
    punct_token_ids = tokenizer.convert_tokens_to_ids(punctuations)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    period_token_id = punct_token_ids[0]
    approved_second_last_token_ids = punct_token_ids + [eos_id, pad_id]

    tokens = tokenizer(sentence, padding="max_length", truncation=True,
                       max_length=max_length, return_tensors="pt")
    # get the mask of the second last tokens
    input_ids = tokens["input_ids"]
    attention_masks = tokens["attention_mask"]
    # print(tokens)
    mask = ~ torch.isin(input_ids[:, -2], torch.tensor(approved_second_last_token_ids))
    input_ids[mask, -2] = period_token_id
    return {"input_ids": input_ids.to(device), "attention_mask": attention_masks.to(device)}


def fill_in_pad_eos_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return tokenizer


def load_encoder_decoder_and_tokenizer(model_name, device):
    # load the target model and tokenizer
    encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_in_pad_eos_token(tokenizer)
    encoder_decoder.resize_token_embeddings(len(tokenizer))

    encoder_decoder = encoder_decoder.to(device)
    return encoder_decoder, tokenizer


def load_source_encoder_and_tokenizer(model_name, device):
    """
    Load source encoder, not as a seq2selm
    :param model_name:
    :param device:
    :return:
    """
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_in_pad_eos_token(tokenizer)
    encoder.resize_token_embeddings(len(tokenizer))

    encoder = encoder.to(device)
    return encoder, tokenizer



def load_tokenizer_models(source_model_name, target_model_name):
    # the goal is to align source model to target model.
    # the target model should be an encoder-decoder model
    # the source model only need embeddings
    source_model = AutoModel.from_pretrained(source_model_name)
    target_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_name)

    source_hidden_dim = source_model.config.hidden_size
    target_hidden_dim = target_model.config.hidden_size

    # https://github.com/huggingface/transformers/pull/24565 legacy, don't use fast
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    # processing tokenizer related paarametrers
    # pad and eos token
    fill_in_pad_eos_token(source_tokenizer)
    fill_in_pad_eos_token(target_tokenizer)
    # resize the tokenizers
    source_model.resize_token_embeddings(len(source_tokenizer))
    target_model.resize_token_embeddings(len(target_tokenizer))

    # move the models to device.
    source_model = source_model.to(device)
    target_model = target_model.to(device)

    return (source_model, target_model,
            source_hidden_dim, target_hidden_dim,
            source_tokenizer, target_tokenizer)


# vec2text/models/model_utils.py
def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def get_embeddings(train_data, test_data,
                   source_model, target_model,
                   source_tokenizer, target_tokenizer,
                   max_length=32, noise_level=0):
    X_tokens = add_punctuation_token_ids(train_data, source_tokenizer, max_length, device)
    Y_tokens = add_punctuation_token_ids(train_data, target_tokenizer, max_length, device)
    X_test_tokens = add_punctuation_token_ids(test_data, source_tokenizer, max_length, device)
    Y_test_tokens = add_punctuation_token_ids(test_data, target_tokenizer, max_length, device)

    with torch.no_grad():
        if source_model.config.is_encoder_decoder:
            print("source model is encoder_decoder")
            X = source_model.encoder(**X_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
            X_test = source_model.encoder(**X_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        else:
            print("source model is not a encoder_decoder")
            X = source_model(**X_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
            X_test = source_model(**X_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y = target_model.encoder(**Y_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y_test = target_model.encoder(**Y_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)

    if noise_level > 0:
        X += noise_level * torch.randn(X.shape)
        X_test += noise_level * torch.rand(X_test.shape)
    return X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens



def get_embeddings_from_encoders(train_data, test_data,
                   source_model, target_model,
                   source_tokenizer, target_tokenizer,
                   max_length=32, noise_level=0):
    X_tokens = add_punctuation_token_ids(train_data, source_tokenizer, max_length, device)
    Y_tokens = add_punctuation_token_ids(train_data, target_tokenizer, max_length, device)
    X_test_tokens = add_punctuation_token_ids(test_data, source_tokenizer, max_length, device)
    Y_test_tokens = add_punctuation_token_ids(test_data, target_tokenizer, max_length, device)

    with torch.no_grad():
        if source_model.config.is_encoder_decoder:
            print("source model is encoder_decoder")
            X = source_model.encoder(**X_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
            X_test = source_model.encoder(**X_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        else:
            print("source model is not a encoder_decoder")
            X = source_model(**X_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
            X_test = source_model(**X_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y = target_model(**Y_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y_test = target_model(**Y_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)

    if noise_level > 0:
        X += noise_level * torch.randn(X.shape)
        X_test += noise_level * torch.rand(X_test.shape)
    return X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens

