import os
import sys

sys.path.append('../src')

from InversionModel import EmbeddingInverter

import torch
import numpy as np
from rouge_score import rouge_scorer
from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1, eps=1e-6)


def load_data(
data_path = "/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus/eng-literal/train.txt"
        ):

    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = [x.replace("\n", "") for x in f.readlines()]
    print(f"Data length {len(data)}")
    data_sampled = data[:10000]+data[-300:]
    return data_sampled


def test_alignment(As, X_test, Y_test):
    x = X_test
    y = Y_test

    xs = []
    for i in range(len(As)):
        x_i = x[:, i, :]
        # print(x_i.shape) # seq x d_s
        a_i = As[i]  # the i th token A
        x_aligned = x_i @ a_i
        xs.append(x_aligned)

    x_ = torch.stack(xs, axis=1)

    return cos(x_, y).mean()


def get_cosine_similarity_aligned(samples, X, Y, X_test, Y_test):
    # see how many samples we need to get proper cosine similarities between test data
    X = X[:samples]
    Y = Y[:samples]
    tokens_nr = X.shape[1]
    # one-off mapping for each token.
    As = []
    Xs_aligned = []
    for i in range(tokens_nr):
        x = X[:, i, :]
        y = Y[:, i, :]
        # print(x.shape, y.shape)
        A = torch.linalg.pinv(x.t() @ x) @ x.t() @ y
        As.append(A)
        x_aligned = x @ A
        Xs_aligned.append(x_aligned)
    Xs = torch.stack(Xs_aligned, axis=1)

    test_cos = test_alignment(As, X_test, Y_test)
    return cos(Xs, Y).mean(), test_cos


def get_embeddings(data, model_g="t5-base", model_s="t5-small", max_length=32,test_samples=300):
    inverter = EmbeddingInverter(model_G_name_or_path=model_g,
                                 model_S_name_or_path=model_s,
                                 max_length=max_length)
    s_embeds, _, _ = inverter.get_embeddings_S(data)
    g_embeds, _, _ = inverter.get_embeddings_G(data)

    X = s_embeds[:-test_samples]
    Y = g_embeds[:-test_samples]
    # assert X.shape[0] == Y.shape[0] == len(data) 

    X_test = s_embeds[-test_samples:]
    Y_test = g_embeds[-test_samples:]
    # assert X_test.shape[0] == Y_test.shape[0] == test_samples

    print(f"X: {X.shape} Y: {Y.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    return X,Y, X_test, Y_test


def main(folder="results/normal_equations/sentence_embeddings"):
    os.makedirs(folder, exist_ok=True)
    # find the optimal sample size first.
    data = load_data()
    best_test_cos = -float("inf")
    patience = 1000
    no_improvements_since = 0

    for max_length in [64, 128]:
        print(f"deal with max length {max_length}")
        f_writer = open(f"{folder}/t5base2small_{max_length}.csv", "a+")
        f_writer.write("max_length,sample_size,X_Y_cos,test_cos\n")
        X, Y, X_test, Y_test = get_embeddings(data, model_g="t5-base", model_s="t5-small", max_length=max_length)

        for sample_size in range(100, 10000, 100):
            X_Y_cos , test_cos = get_cosine_similarity_aligned(samples=sample_size, X=X, Y=Y, X_test=X_test, Y_test=Y_test)
            X_Y_cos = X_Y_cos.detach().cpu().numpy()
            test_cos = test_cos.detach().cpu().numpy()
            print(f"max length: {max_length}, sample size: {sample_size},X-Y-Cos: {X_Y_cos}, test-cos: {test_cos}\n")
            f_writer.write(f"{max_length},{sample_size},{X_Y_cos},{test_cos}\n")

            if sample_size > 1000:
                if test_cos > best_test_cos:
                    best_test_cos = test_cos
                    no_improvements_since = 0
                else:
                    no_improvements_since += 100

                if no_improvements_since >= patience:
                    print("Early stopping triggered due to no improvement in test cosine similarity")
                    break

        f_writer.close()

    # TODO:
    # check how many dimensions are necessary to invert


if __name__ == '__main__':
    main()



