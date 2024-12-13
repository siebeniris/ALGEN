import torch

cos = torch.nn.CosineSimilarity(dim=1)


def mapping_X_to_Y(X, Y):
    # mappting with normal equation.
    batch_size, seq_len, _ = X.shape
    X_ = X.reshape(-1, X.shape[-1])
    Y_ = Y.reshape(-1, Y.shape[-1])
    As = torch.linalg.pinv(X_.T @ X_) @ X_.T @ Y_
    Xs = X_ @ As

    Xs_ = Xs.view(batch_size, seq_len, Xs.shape[-1])
    return cos(Xs, Y_).mean(), Xs_, As



def test_alignment(X_test, Y_test, As):
    # using As from normal equation
    test_batch_size, test_seq_len, test_x_dim = X_test.shape

    x = X_test.reshape(-1, X_test.shape[-1])
    y = Y_test.reshape(-1, Y_test.shape[-1])
    x_ = x @ As
    return cos(x_, y).mean(), x_.view(test_batch_size, test_seq_len, x_.shape[-1])

