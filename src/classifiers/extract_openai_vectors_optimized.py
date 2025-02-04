import os
import random
import torch
import datasets
import tiktoken
import asyncio
import aiohttp
from tqdm import tqdm
from src.classifiers.data_helper import save_embeddings

import time

API_KEY = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")



def get_texts(dataset):
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]
    val_texts = dataset["dev"]["text"]
    return train_texts, val_texts, test_texts


def get_vectors(texts, model):
    BATCH_SIZE = 64
    MAX_CONCURRENT_REQUESTS = 50  # Controls parallel requests
    RATE_LIMIT_RPM = 2900  # OpenAI limit
    REQUEST_INTERVAL = 60 / RATE_LIMIT_RPM  # Minimum delay between requests
    TIMEOUT = 30  # Max time in seconds for a request

    async def fetch_embedding(session, batch, semaphore):
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "input": batch}

        async with semaphore:  # Limits concurrent requests
            for attempt in range(5):  # Retry up to 5 times
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=TIMEOUT) as response:
                        if response.status == 200:
                            data = await response.json()
                            return [entry["embedding"] for entry in data["data"]]
                        elif response.status == 429:  # Handle rate limits
                            wait_time = min(2 ** attempt + random.uniform(0, 1), 30)  # Exponential backoff (max 30s)
                            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"Error: {response.status} - {await response.text()}")
                            return None  # Skip on failure
                except asyncio.TimeoutError:
                    print(f"Timeout occurred for batch, skipping...")
                    return None
        return None  # If all retries fail

    # Main async processing function
    async def process_texts(texts):
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Limits concurrency
        tasks = []

        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(texts), BATCH_SIZE)):
                batch = texts[i:i + BATCH_SIZE]
                task = asyncio.create_task(fetch_embedding(session, batch, semaphore))  # Non-blocking tasks
                tasks.append(task)
                await asyncio.sleep(REQUEST_INTERVAL)  # Respect rate limit

            # Run requests in parallel with timeout handling
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return [torch.tensor(embedding) for batch in results if batch for embedding in batch]

    start_time = time.time()
    embeddings = asyncio.run(process_texts(texts))

    end_time = time.time()
    print(f"Finished processing all embeddings in {end_time - start_time:.2f} seconds.")

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
    # get labels
    train_labels = torch.tensor(dataset["train"]["label"])
    dev_labels = torch.tensor(dataset["dev"]["label"])
    test_labels = torch.tensor(dataset["test"]["label"])


    print("Getting dataset texts vectors")


    train_vectors = get_vectors(train_texts, model_name)
    print(train_vectors)
    print(f"saving train embeddings and labels to {embedding_dir}")

    save_embeddings(train_vectors, train_labels, embedding_dir, "train")

    dev_vectors = get_vectors(dev_texts, model_name)
    print(f"saving dev embeddings and labels to {embedding_dir}")
    save_embeddings(dev_vectors, dev_labels, embedding_dir, "dev")

    test_vectors = get_vectors(test_texts, model_name)
    print(f"saving test embeddings and labels to {embedding_dir}")
    save_embeddings(test_vectors, test_labels, embedding_dir, "test")

    print(
        f"embeddings shape: train {train_vectors.shape}, dev {dev_vectors.shape}, test {test_vectors.shape}")


if __name__ == '__main__':
    gpt_embedders = ["text-embedding-ada-002"]
    datasets_names = ["yiyic/s140_ds"]

    for gpt_embedder in gpt_embedders:
        for dataset_name in datasets_names:
            print(f"extracting embeddings from {gpt_embedder} with dataset {dataset_name}")
            extract_vectors_per_dataset(gpt_embedder, dataset_name, max_length=32, data_dir="datasets/")

