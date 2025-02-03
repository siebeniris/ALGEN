import torch


# shuffling embeddings as a defense, that only target embeddings are shuffled,
# the attack embeddings cannot be shuffled. but the

def shuffle_embeddings(X, X_val, X_test, Y_test_attention):
    sample_nr, X_dim = X.shape
    perm = torch.randperm(X_dim).to(X.device)

    X_shuffled = X[:, perm]
    X_shuffled_norm = X_shuffled / torch.norm(X_shuffled, p=2, dim=1, keepdim=True)

    X_val_shuffled = X_val[:, perm]
    X_val_shuffled_norm = X_val_shuffled / torch.norm(X_val_shuffled, p=2, dim=1, keepdim=True)

    X_test_shuffled = X_test[:, perm]
    X_test_shuffled_norm = X_test_shuffled / torch.norm(X_test_shuffled, p=2, dim=1, keepdim=True)

    Y_test_attention = Y_test_attention[:, perm]
    return X_shuffled_norm, X_val_shuffled_norm, X_test_shuffled_norm, Y_test_attention


def shuffle_only_embeddings(X, X_val, X_test):
    sample_nr, X_dim = X.shape
    perm = torch.randperm(X_dim).to(X.device)

    X_shuffled = X[:, perm]
    X_shuffled_norm = X_shuffled / torch.norm(X_shuffled, p=2, dim=1, keepdim=True)

    X_val_shuffled = X_val[:, perm]
    X_val_shuffled_norm = X_val_shuffled / torch.norm(X_val_shuffled, p=2, dim=1, keepdim=True)

    X_test_shuffled = X_test[:, perm]
    X_test_shuffled_norm = X_test_shuffled / torch.norm(X_test_shuffled, p=2, dim=1, keepdim=True)

    return X_shuffled_norm, X_val_shuffled_norm, X_test_shuffled_norm



def shuffling_one_set_embeddings(X, Y_test_attention):
    ample_nr, X_dim = X.shape
    perm = torch.randperm(X_dim).to(X.device)

    X_shuffled = X[:, perm]
    X_shuffled_norm = X_shuffled / torch.norm(X_shuffled, p=2, dim=1, keepdim=True)

    Y_test_attention = Y_test_attention[:, perm]
    return X_shuffled_norm, Y_test_attention
