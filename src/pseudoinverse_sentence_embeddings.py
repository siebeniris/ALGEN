import os
import sys

sys.path.append('../src')

import itertools
from InversionModel import EmbeddingInverter

import torch
import numpy as np
from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1, eps=1e-6)


def test_alignment( X_test, Y_test, As, slice_x=None, shift="left"):
    x = X_test
    y = Y_test
    test_batch_size, test_seq_len, test_x_dim = x.shape

    x = x.reshape(-1, x.shape[-1])
    # print("test...", x.shape, y.shape, As.shape)

    if shift == "left":
        x = x[:,:slice_x]
    elif shift == "right":
        x = x[:, slice_x:]
    else:
        x = x

    x_ = x @ As

    # xs = []
    # for i in range(len(As)):
    #     if shift == "left":
    #         x_i = x[:, i, :slice_x]
    #     elif shift == "right":
    #         x_i = x[:, i, slice_x:]
    #     else:
    #         x_i = x[:, i, :]
    #
    #     # print(x_i.shape) # seq x d_s
    #     a_i = As[i]  # the i th token A
    #     x_aligned = x_i @ a_i
    #     xs.append(x_aligned)
    #
    # x_ = torch.stack(xs, axis=1)

    x_ = x_.view(test_batch_size, test_seq_len, x_.shape[-1])
    return cos(x_, y).mean(), x_


def get_cosine_similarity_aligned(samples, X, Y, X_test, Y_test, slice_x=None, shift="left"):
    # see how many samples we need to get proper cosine similarities between test data
    X = X[:samples]
    Y_ = Y[:samples]
    tokens_nr = X.shape[1]
    batch_size, seq_len, x_dim = X.shape
    X = X.reshape(-1, X.shape[-1])
    Ys = Y_.reshape(-1, Y.shape[-1])

    if shift == "left":
        X = X[:,:slice_x]
    elif shift == "right":
        X = X[:, slice_x:]
    else:
        X = X
    print(X.shape, Ys.shape)
    # one-off mapping for each token.
    As = []
    Xs_aligned = []
    # for i in range(tokens_nr):
    #     if shift == "left":
    #         x = X[:, i, :slice_x]
    #     elif shift == "right":
    #         x = X[:, i, slice_x:]
    #     else:
    #         x = X[:, i, :]
    #     # x = X[:, i, :]
    #     y = Y[:, i, :]
    #     # print(x.shape, y.shape)
    #     A = torch.linalg.pinv(x.t() @ x) @ x.t() @ y
    #     As.append(A)
    #     x_aligned = x @ A
    #     Xs_aligned.append(x_aligned)
    # Xs = torch.stack(Xs_aligned, axis=1)
    # print(X.shape, Y.shape)

    As = torch.linalg.pinv(X.T @ X) @ X.T @ Ys
    Xs = X @ As
    # print(X.shape, Y.shape, As.shape)

    test_cos, x_ = test_alignment(X_test, Y_test, As, slice_x, shift)
    # print(cos(Xs, Y).mean(), test_cos)
    Xs = Xs.view(batch_size, seq_len, Xs.shape[-1])
    return cos(Xs, Y_).mean(), test_cos


def find_optimal_dimension(X, Y, X_test, Y_test, sample_size, writer):
    dim = X.shape[-1]
    for slice_x in range(5, dim, 5):
        for shift in ["left", "right"]:
            # for shift in ["random"]:
            X_Y_cossim, test_cosine_sim = get_cosine_similarity_aligned(samples=sample_size,
                                                                        X=X, Y=Y,
                                                                        X_test=X_test,
                                                                        Y_test=Y_test,
                                                                        slice_x=slice_x,
                                                                        shift=shift)
            print(f"X size {sample_size}, shift {shift}, dim {slice_x}, "
                  f"test cossim: {test_cosine_sim}, "
                  f"train cossim: {X_Y_cossim}")
            if shift == "right":
                dim_size = dim - slice_x
            else:
                dim_size = slice_x
            writer.write(f"{sample_size},{shift},{dim_size},{test_cosine_sim},{X_Y_cossim}\n")


def main(folder="sentence_embeddings/eng_latn/"):
    os.makedirs(folder, exist_ok=True)
    # find the optimal sample size first.
    model_names = ["t5-small", "t5-base"]
    max_lengths = [32, 64, 128]

    outputfolder = "results/normal_equations/sentence_embeddings"

    for p in itertools.permutations(model_names):
        s, t = p
        for max_length in max_lengths:
            print(f"processing mapping {s} to {t} max length {max_length}")
            X_file = f"{folder}/{s}_train_{max_length}.npy"
            Y_file = f"{folder}/{s}_test_{max_length}.npy"
            outputfile = f"{outputfolder}/{s}_{t}_{max_length}_samples1000_global.csv"
            # outputfile = f"{outputfolder}/{s}_{t}_{max_length}_global.csv"

            if not os.path.exists(outputfile):
                if os.path.exists(X_file) and os.path.exists(Y_file):
                    X = np.load(f"{folder}/{s}_train_{max_length}.npy")
                    X_test = np.load(f"{folder}/{s}_test_{max_length}.npy")

                    Y = np.load(f"{folder}/{t}_train_{max_length}.npy")
                    Y_test = np.load(f"{folder}/{t}_test_{max_length}.npy")

                    X = torch.from_numpy(X)
                    Y = torch.from_numpy(Y)
                    X_test = torch.from_numpy(X_test)
                    Y_test = torch.from_numpy(Y_test)

                    best_test_cos = -float("inf")
                    patience = 1000
                    no_improvements_since = 0

                    ### output writer.
                    f_writer = open(outputfile, "a+")
                    find_optimal_dimension(X, Y, X_test, Y_test, 1000, f_writer)

                    # for sample_size in range(100, 5000, 100):
                    #     X_Y_cos, test_cos = get_cosine_similarity_aligned(samples=sample_size,
                    #                                                       X=X, Y=Y,
                    #                                                       X_test=X_test,
                    #                                                       Y_test=Y_test)
                    #     X_Y_cos = X_Y_cos.numpy()
                    #     test_cos = test_cos.numpy()
                    #     print(
                    #         f"max length: {max_length}, sample size: {sample_size},X-Y-Cos: {X_Y_cos}, test-cos: {test_cos}\n")
                    #     f_writer.write(f"{max_length},{sample_size},{X_Y_cos},{test_cos}\n")
                    #
                    #     if sample_size > 1000:
                    #         if test_cos > best_test_cos:
                    #             best_test_cos = test_cos
                    #             no_improvements_since = 0
                    #         else:
                    #             no_improvements_since += 100
                    #
                    #         if no_improvements_since >= patience:
                    #             print("Early stopping triggered due to no improvement in test cosine similarity")
                    #             break

                    f_writer.close()


if __name__ == '__main__':
    main()
