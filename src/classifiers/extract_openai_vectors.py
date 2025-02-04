import os

import torch
import datasets
import tiktoken
import backoff
from tqdm import tqdm
from openai import OpenAI
import openai

from src.classifiers.data_helper import save_embeddings

import time

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


def get_texts(dataset):
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]
    val_texts = dataset["dev"]["text"]
    return train_texts, val_texts, test_texts


def get_vectors(texts, model):
    BATCH_SIZE = 64
    MAX_REQUESTS_PER_MINUTE = 2900  # Keeping it below 3000 RPM
    REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Time between requests

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def get_embedding_batch(batch):
        response = openai.embeddings.create(
            model=model,
            encoding_format="float",
            input=batch
        )
        return [torch.tensor(d.embedding) for d in response.data]

    # Process in batches
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]

        try:
            embeddings.extend(get_embedding_batch(batch))
        except openai.OpenAIError as e:
            print(f"Error on batch {i}: {e}")
            continue  # Skip this batch if it fails

        # Respect rate limits
        time.sleep(REQUEST_INTERVAL)

    # Now embeddings contain all vectors
    print("Finished processing all embeddings.")
    return torch.stack(embeddings, dim=0)


def extract_vectors_per_dataset(model_name, dataset_name, max_length=32, data_dir="datasets/"):
    # encoder
    encoder = tiktoken.encoding_for_model(model_name)

    def truncate_text(example, max_length):
        tokens = encoder.encode(example["text"])
        truncated_tokens = tokens[:max_length]
        example["text"] = encoder.decode(truncated_tokens)
        return example

    model_name_ = model_name.replace("/", "__")
    dataset_name_ = dataset_name.replace("/", "__")

    embedding_dir = os.path.join(data_dir, f"{model_name_}_{dataset_name_}_NoDefense")
    os.makedirs(embedding_dir, exist_ok=True)
    dataset = datasets.load_dataset(dataset_name)

    print("Truncating texts ...")
    dataset = dataset.map(lambda x: truncate_text(x, max_length=max_length))

    train_texts, dev_texts, test_texts = get_texts(dataset)

    print("Loading dataset texts")
    train_vectors = get_vectors(train_texts, model_name)
    dev_vectors = get_vectors(dev_texts, model_name)
    test_vectors = get_vectors(test_texts, model_name)

    # get labels
    train_labels = torch.tensor(dataset["train"]["label"])
    dev_labels = torch.tensor(dataset["dev"]["label"])
    test_labels = torch.tensor(dataset["test"]["label"])

    print(
        f"embeddings shape: train {train_vectors.shape}, dev {dev_vectors.shape}, test {test_vectors.shape}")

    print(f"saving embeddings and labels to {embedding_dir}")
    save_embeddings(train_vectors, train_labels, embedding_dir, "train")
    save_embeddings(dev_vectors, dev_labels, embedding_dir, "dev")
    save_embeddings(test_vectors, test_labels, embedding_dir, "test")


if __name__ == '__main__':
    gpt_embedders = ["text-embedding-ada-002"]
    # datasets_names = ["yiyic/snli_ds", "yiyic/sst2_ds", "yiyic/s140_ds"]
    datasets_names = ["yiyic/snli_ds", "yiyic/sst2_ds"]
    # extract_vectors_per_dataset(gpt_embedders[0], datasets_names[0], max_length=32, data_dir="datasets/")


    for gpt_embedder in gpt_embedders:
        for dataset_name in datasets_names:
            print(f"extracting embeddings from {gpt_embedder} with dataset {dataset_name}")
            extract_vectors_per_dataset(gpt_embedder, dataset_name, max_length=32, data_dir="datasets/")

