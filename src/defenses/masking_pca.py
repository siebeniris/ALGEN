import torch
from fancyimpute import IterativeSVD


def masking_embeddings(X):
    X_mask = X.clone()
    X_mask[:,0] = 1.
    X_masked_norm = X_mask / torch.norm(X_mask, p=2, dim=1, keepdim=True)
    return X_masked_norm


