import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss, CosineEmbeddingLoss
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from eval_metrics import get_rouge_scores, loss_metrics

from inversion_utils import (
    load_tokenizer_models,
    get_embeddings,
)
from inversion_methods import mapping_X_to_Y, test_alignment
from eval_metrics import eval_decoding
from utils import get_device, pairwise_cosine
from AlignerOT import TokenAlignerOT
import random

################################################################
# Alignment: 1, align the last dimension using normal equation first
# 2: align the tokens using optimal transport with trainable parameters, doesn't seem to work.


cos = torch.nn.CosineSimilarity(dim=1)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def in_debug_mode():
    try:
        import pydevd  # PyCharm debugger
        return True
    except ImportError:
        return False


# hyperparameter initialization
max_length = 32
source_model_name = "t5-base"
target_model_name = "google/flan-t5-small"
lang_data_dir = "/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus"
lang_script = "eng-literal"
train_samples = 500
learning_rate = 1e-4
epochs = 100
device = get_device()

# load model
source_model, target_model, source_hidden_dim, target_hidden_dim, source_tokenizer, target_tokenizer \
    = load_tokenizer_models(source_model_name, target_model_name)

# load data
lang_data_dir_ = os.path.join(lang_data_dir, lang_script)
with open(os.path.join(lang_data_dir_, "train.txt")) as f:
    train_data = [x.replace("\n", "") for x in f.readlines()]

with open(os.path.join(lang_data_dir_, "test.txt")) as f:
    test_data = [x.replace("\n", "") for x in f.readlines()][:200]

train = train_data[:train_samples]
test = test_data

print(f"test data {len(test)}, train data {len(train)}")

# Get embeddings.
X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens = \
    get_embeddings(train, test, source_model, target_model,
                   source_tokenizer, target_tokenizer, max_length)

print("shape of X:", X.shape, "Y:", Y.shape, "X_test:", X_test.shape, "Y_test:", Y_test.shape)


# Y test gold
Y_test_gold = [target_tokenizer.decode(tb, skip_special_tokens=True)
            for tb in Y_test_tokens["input_ids"]]


# normal equation, align the last dimension.
# source_hidden_dim to target_hidden_dim.
X_y_cos, X_aligned, As = mapping_X_to_Y(X, Y)
x_y_test_cos, x_test_aligned = test_alignment(X_test, Y_test, As)
print("result aligning embeddings using normal equation:")
print(f"X and Y cosine {X_y_cos} and X_test and Y_test: {x_y_test_cos}")

# get attention masks.
X_attention_mask = X_tokens["attention_mask"]
Y_attention_mask = Y_tokens["attention_mask"]
X_test_attention_mask = X_test_tokens["attention_mask"]
Y_test_attention_mask = Y_test_tokens["attention_mask"]

source_seq_len = X_aligned.shape[1]
target_seq_len = Y.shape[1]

# initialize aligner, to align tokens.
aligner = TokenAlignerOT(source_seq_len, target_seq_len, device)
aligner = aligner.to(device)

optimizer = torch.optim.Adam(aligner.parameters(), lr=learning_rate)


def eval_epoch(aligner, X_test, Y_test, Y_test_attention_mask, Y_test_gold):
    aligner.eval()

    cos_loss = 0
    mse_loss = 0
    cosine_list = []

    batch_size = X_test.shape[0]

    Xs_aligned = []

    with torch.no_grad():
        for i in tqdm(range(X_test.shape[0]), desc="Evaluating"):
            x_i = X_test[i]
            y_i = Y_test[i]
            x_i_aligned = aligner(x_i, y_i)

            cosine_list.append(cos(x_i_aligned, y_i).mean().detach().cpu().numpy())

            cosloss, mseloss = loss_metrics(x_i_aligned, y_i)

            cos_loss += cosloss.item()
            mse_loss += mseloss.item()

            Xs_aligned.append(x_i_aligned)

    # decoding evaluation.
    X_aligned = torch.stack(Xs_aligned, dim=0)

    rouge_result_dict, X_test_output, Y_test_output = eval_decoding(
        X_aligned, Y_test, Y_test_gold, Y_test_attention_mask,
        target_model,
        target_tokenizer,
        num_beams=3,
        do_sample=False,
        repetition_penalty=2.0,
        length_penalty=2.0,
        top_k=None, top_p=None, temperature=None,
        max_length=max_length
    )

    df_output = pd.DataFrame({
        "X_output": X_test_output,
        "Y_output": Y_test_output,
        "Y_gold": Y_test_gold
    })

    eval_metrics = {
        "cos_loss": cos_loss / batch_size,
        "mse_loss": mse_loss / batch_size,
        "cos_sims": np.mean(cosine_list),

    }
    print(eval_metrics)
    print(rouge_result_dict)

    # print(df_output.head(50))


def train_epoch(aligner, epochs, X, Y, X_test, Y_test, Y_test_attention_mask, Y_test_gold):
    batch_size = X.shape[0]

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        aligner.train()
        cosine_loss = 0
        mse_loss = 0

        cosine_sims = []
        for i in tqdm(range(batch_size), desc="Training"):
            X_i = X[i]
            Y_i = Y[i]

            X_i = X_i.to(device)
            Y_i = Y_i.to(device)
            # TODO: should I use only masked parts?
            X_i_aligned = aligner(X_i, Y_i)

            # cosine similarity and aligned X.
            X_y_cos = cos(X_i_aligned, Y_i).mean().detach().cpu().numpy()

            cosine_sims.append(X_y_cos)

            cosloss, mseloss = loss_metrics(X_i_aligned, Y_i)

            # backpropagation
            optimizer.zero_grad()
            cosloss.backward()
            optimizer.step()

            cosine_loss += cosloss.item()
            mse_loss += mseloss.item()


        train_metrics = {
            "cos_loss": cosine_loss/batch_size,
            "mse_loss": mse_loss/batch_size,
            "cos_sims": np.mean(cosine_sims),

        }
        print("training metrics:", train_metrics)
        # evaluate
        eval_epoch(aligner, X_test, Y_test, Y_test_attention_mask, Y_test_gold)


if __name__ == '__main__':

    train_epoch(aligner, epochs, X_aligned, Y, x_test_aligned, Y_test, Y_test_attention_mask, Y_test_gold)

