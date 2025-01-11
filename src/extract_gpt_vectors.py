import os

import numpy as np
import torch
import datasets
from transformers import AutoTokenizer
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import time

from inversion_utils import fill_in_pad_eos_token, add_punctuation_token_ids

device = "cuda" if torch.cuda.is_available() else "cpu"


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


def get_texts(data_name):
    """ Load the dataset from huggingface."""
    dataset = datasets.load_dataset(data_name)
    train_texts = dataset["train"]["text"][:1000]
    test_texts = dataset["test"]["text"][:200]
    val_texts = dataset["dev"]["text"][:200]
    return train_texts, val_texts, test_texts


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_in_pad_eos_token(tokenizer)
    return tokenizer


def get_Y_embeddings_from_encoders(train_data, val_data, test_data, target_tokenizer, max_length=32):
    Y_tokens = add_punctuation_token_ids(train_data, target_tokenizer, max_length, device)
    Y_val_tokens = add_punctuation_token_ids(val_data, target_tokenizer, max_length, device)
    Y_test_tokens = add_punctuation_token_ids(test_data, target_tokenizer, max_length, device)
    return Y_tokens, Y_val_tokens, Y_test_tokens

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def retrieve_openai_vector(texts, model):
    response = client.embeddings.create(input=texts, model=model, encoding_format="float")

    vectors = []
    for d in response.data:
        vec = d.embedding
        vectors.append(np.array(vec))
    vectors_ = np.stack(vectors, axis=0)
    return vectors_


def get_vectors_gpt_per_model_dataset(
        tokenizer,
        gpt_embedder="text-embedding-ada-002",
        max_length=32,
        attack_model="google/flan-t5-small", dataset="yiyic/mmarco_english",
        outputdir="datasets/vectors"):
    """
    Get the look-up tokens.
    :param attack_model:
    :return:
    """
    # get the path to output directory
    outputdir = os.path.join(outputdir, gpt_embedder)
    outputdir = os.path.join(outputdir, attack_model.replace("/", "_"))
    dataset_name = dataset.replace("/", "_")
    outputdir = os.path.join(outputdir, dataset_name)

    print(f"outputdir :{outputdir}")
    os.makedirs(outputdir, exist_ok=True)

    train_texts, val_texts, test_texts = get_texts(dataset)

    Y_tokens, Y_val_tokens, Y_test_tokens = get_Y_embeddings_from_encoders(train_texts, val_texts, test_texts,
                                                                        tokenizer,
                                                                        max_length=max_length)

    # get the ground truth texts.
    true_train_texts = tokenizer.batch_decode(Y_tokens["input_ids"], skip_special_tokens=True)
    true_val_texts = tokenizer.batch_decode(Y_val_tokens["input_ids"], skip_special_tokens=True)
    true_test_texts = tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True)
    print(f"train data {len(true_train_texts)}, val {len(true_val_texts)}, test {len(true_test_texts)}")

    # get the vectors.
    train_vectors = retrieve_openai_vector(true_train_texts, gpt_embedder)
    val_vectors = retrieve_openai_vector(true_val_texts, gpt_embedder)
    test_vectors = retrieve_openai_vector(true_test_texts, gpt_embedder)

    print(f"vectors shape: train {train_vectors.shape}, dev {val_vectors.shape}, test {test_vectors.shape}")
    save_path = os.path.join(outputdir, f"vecs_maxlength{max_length}.npz")
    print(f"saving vectors to {save_path}")
    np.savez_compressed(save_path, train=train_vectors, dev=val_vectors, test=test_vectors)


if __name__ == '__main__':

    attacker_models = ["google/flan-t5-small", "google/umt5-small", "google/mt5-small"]
    max_length = 32

    # dataset_names = ["yiyic/mmarco_english", "yiyic/mmarco_french", "yiyic/mmarco_spanish", "yiyic/mmarco_german"]
    # datasets_extra = ["yiyic/mmarco_chinese", "yiyic/mmarco_vietnamese"]
    dataset_names = ["yiyic/multiHPLT_english"]
    datasets_extra = dataset_names
    gpt_embedders = ["text-embedding-ada-002", "text-embedding-3-large"]
    # gpt_embedders = ["text-embedding-3-large"]
    for attacker_model in attacker_models:
        tokenizer = load_tokenizer(model_name=attacker_model)
        for gpt_embedder in gpt_embedders:
            for dataset in dataset_names:
                print(f"getting vectors for {attacker_model} dataset {dataset} with {gpt_embedder}")
                get_vectors_gpt_per_model_dataset(tokenizer, gpt_embedder, max_length, attacker_model, dataset)
                time.sleep(5)
            if "flan" not in attacker_model:
                for dataset in datasets_extra:
                    print(f"getting vectors for {attacker_model} dataset {dataset} with {gpt_embedder}")
                    get_vectors_gpt_per_model_dataset(tokenizer, gpt_embedder, max_length, attacker_model, dataset)





