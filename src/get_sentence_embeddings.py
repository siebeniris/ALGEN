import os

import plac
import torch


from transformers import T5Tokenizer, T5Model
from torch.nn import LayerNorm
import transformers.models.t5.modeling_t5 as t5_modeling
import numpy as np
t5_modeling.T5LayerNorm = LayerNorm
device = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm

output_dir = "sentence_embeddings/eng_latn"
os.makedirs(output_dir, exist_ok=True)

def get_tokenizer_encoder(model_name="t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5Model.from_pretrained(model_name).encoder
    return tokenizer, encoder


def get_embeddings(tokenizer, encoder, data_batch, max_length=128):
    inputs = tokenizer(
        data_batch,
        return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
    ).to(device)
    return encoder(**inputs).last_hidden_state


def load_data(
data_path = "/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus/eng-literal/train.txt"
        ):

    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = [x.replace("\n", "") for x in f.readlines()]
    print(f"Data length {len(data)}")
    # data_sampled = data[:10000]+data[-300:]
    return data[:5000], data[-300:]


def get_embeddings_stack(model_name, data, max_length):
    tokenizer, encoder = get_tokenizer_encoder(model_name)
    embeddings_list = []
    batch_size = 50
    num_batches = int(np.ceil(len(data) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        train_data_batch = data[start_idx:end_idx]
        embeddings_batch = get_embeddings(tokenizer, encoder, train_data_batch, max_length)
        embeddings_list.append(embeddings_batch)
    embeddings = torch.cat(embeddings_list, dim=0)

    return embeddings.detach().cpu().numpy()



def main(data_path):
    train_data, test_data = load_data(data_path)
    for model_name in ["t5-small", "t5-base", "google/flan-t5-base", "google/flan-t5-base" ]:
        for max_length in [32, 64, 128]:
            print(f"processing {model_name} max length {max_length}")
            train_embeddings = get_embeddings_stack(model_name, train_data, max_length=max_length)
            np.save(f"{output_dir}/{model_name}_train_{max_length}.npy", train_embeddings)

            test_embeddings = get_embeddings_stack(model_name, test_data, max_length=max_length)
            np.save(f"{output_dir}/{model_name}_test_{max_length}.npy", test_embeddings)


if __name__ == '__main__':
    import plac
    plac.call(main)