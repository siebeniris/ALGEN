import torch


def shuffle_embeddings(X):
    sample_nr, X_dim = X.shape
    perm = torch.randperm(X_dim)
    X_shuffled = X[:, perm]
    X_shuffled_norm = X_shuffled / torch.norm(X_shuffled, p=2, dim=1, keepdim=True)
    return X_shuffled_norm
