import os

import torch
import datasets
import tiktoken
import asyncio
import aiohttp
from tqdm import tqdm
from src.classifiers.data_helper import save_embeddings
import numpy as np
import time

API_KEY = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")



def get_texts(dataset):
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]
    val_texts = dataset["dev"]["text"]
    return train_texts, val_texts, test_texts


def get_vectors(texts, model):
    BATCH_SIZE = 64  # Max texts per request (fits in 2048 tokens)
    MAX_CONCURRENT_REQUESTS = 50  # Controls parallel requests
    RATE_LIMIT_RPM = 2900  # Stay under OpenAI limit
    REQUEST_INTERVAL = 60 / RATE_LIMIT_RPM  # Minimum delay between requests

    # OpenAI API request function (async)
    async def fetch_embedding(session, batch, semaphore):
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "text-embedding-ada-002", "input": batch}

        async with semaphore:  # Limits concurrent requests
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return [entry["embedding"] for entry in data["data"]]
                elif response.status == 429:  # Handle rate limits
                    await asyncio.sleep(1)  # Wait & retry
                    return await fetch_embedding(session, batch, semaphore)
                else:
                    print(f"Error: {response.status}")
                    return None  # Skip on failure

    # Main async processing function
    async def process_texts():
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Limits concurrency
        tasks = []

        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(texts), BATCH_SIZE)):
                batch = texts[i:i + BATCH_SIZE]
                tasks.append(fetch_embedding(session, batch, semaphore))
                await asyncio.sleep(REQUEST_INTERVAL)  # Stay under API rate limit

            # Run requests in parallel
            results = await asyncio.gather(*tasks)

        return [torch.tensor(embedding) for batch in results if batch for embedding in batch]
    # Now embeddings contain all vectors

    start_time = time.time()
    embeddings = asyncio.run(process_texts())


    end_time = time.time()
    print(f"Finished processing all embeddings in {end_time - start_time:.2f} seconds.")

    return torch.stack(embeddings, dim=0)
def chunk_list(texts, num_chunks=10):
    """Splits a list into num_chunks parts as evenly as possible."""
    return np.array_split(texts, num_chunks)

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

    train_texts_chunks = chunk_list(train_texts, num_chunks=5)

    print("Loading dataset texts")

    train_vectors_ls = []
    for chunk in tqdm(train_texts_chunks):
        vectors = get_vectors(list(chunk), model_name)
        train_vectors_ls.append(vectors)
    train_vectors = torch.cat(train_vectors_ls, dim=0)

    dev_vectors = get_vectors(dev_texts, model_name)
    test_vectors = get_vectors(test_texts, model_name)
    print(train_vectors)

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
    datasets_names = ["yiyic/s140_ds"]

    for gpt_embedder in gpt_embedders:
        for dataset_name in datasets_names:
            print(f"extracting embeddings from {gpt_embedder} with dataset {dataset_name}")
            extract_vectors_per_dataset(gpt_embedder, dataset_name, max_length=32, data_dir="datasets/")

