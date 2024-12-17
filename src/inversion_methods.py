import torch
import numpy as np
import ot
from ot.gromov import gromov_wasserstein
from utils import pairwise_cosine

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


########################################################################

def optimal_transport_weight(sim, device,
                             C1=None, C2=None,
                             reg=0.1, reg_m=0.01,
                             ot_strategy="ub_sinkhorn"):
    # n x m tokenization.
    # the tokens are not padded.
    M = - sim.data.cpu().numpy()
    if C1 is not None:
        # print("ha")
        C1 = - C1.data.cpu().numpy()
        C2 = - C2.data.cpu().numpy()

    n, m = M.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    if ot_strategy == "ub_sinkhorn":
        weight = torch.FloatTensor(ot.unbalanced.sinkhorn_unbalanced(a, b,
                                                                    M, reg, reg_m))
    elif ot_strategy == "gw":
        weight = torch.FloatTensor(gromov_wasserstein(C1, C2, a, b,
                                                    'kl_loss'))
    else:
        weight = torch.FloatTensor(ot.emd(a, b, M))
    weight = weight.to(device)
    return weight


def optimal_transport_align(X, Y, device,
                            reg=0.1, reg_m=0.01,
                            ot_strategy="ub_sinkhorn"):
    batch_size = X.shape[0]
    cos_sims = []
    X_aligned_ot = []
    Ts = []

    for i in range(batch_size):
        X_i = X[i]
        Y_i = Y[i]

        A = pairwise_cosine(X_i, Y_i)
        if ot_strategy == "gw":
            C1 = pairwise_cosine(X_i, X_i)
            C2 = pairwise_cosine(Y_i, Y_i)

            W = optimal_transport_weight(A, device, C1, C2, ot_strategy="gw")
        else:
            W = optimal_transport_weight(A, device,
                                        reg=reg,
                                        reg_m=reg_m,
                                        ot_strategy=ot_strategy)
        T = A * W
        # print(W.shape, A.shape, T.shape) # 32x32
        Ts.append(T)

        # 16, 512,
        X_i_aligned = torch.mm(T.t(), X_i)
        cos_sims.append(cos(X_i_aligned, Y_i).mean().detach().cpu().numpy())
        X_aligned_ot.append(X_i_aligned)

    Xs = torch.stack(X_aligned_ot, dim=0)
    Ts_stack = torch.stack(Ts, dim=0)
    X_Y_cos = np.mean(cos_sims)
    return X_Y_cos, Xs, Ts_stack


def optimal_transport_align_test(X_test, Y_test, T):
    cosine_list = []
    x_test_aligned_list = []

    for i in range(X_test.shape[0]):

        x_test_i = X_test[i]
        y_test_i = Y_test[i]

        x_test_aligned_i = torch.mm(T, x_test_i)
        x_test_aligned_list.append(x_test_aligned_i)
        cosine_list.append(cos(x_test_aligned_i, y_test_i).mean().detach().cpu().numpy())

    return np.mean(cosine_list), torch.stack(x_test_aligned_list, dim=0)


def test_alignment_ot(X_aligned, Y, device, ot_strategy="ub_sinkhorn"):
    cos_sims = []
    X_aligned_ot = []
    Ts = []
    for i in range(X_aligned.shape[0]):
        X_i = X_aligned[i]
        Y_i = Y[i]

        # Ablation studied: with mask is not better.
        # X_attention_mask_i = X_attention_mask[i]
        # Y_attention_mask_i = Y_attention_mask[i]
        #
        # X_i_masked = X_i[X_attention_mask_i.bool(), :]
        # Y_i_masked = Y_i[Y_attention_mask_i.bool(), :]
        # print(X_i_masked.shape, Y_i_masked.shape)

        A = pairwise_cosine(X_i, Y_i)
        W = optimal_transport_weight(A, device=device, ot_strategy=ot_strategy)
        T = A * W
        Ts.append(T)
        # 16, 512,
        X_i_aligned = torch.mm(T.t(), X_i)

        cos_sims.append(cos(X_i_aligned, Y_i).mean().detach().cpu().numpy())
        X_aligned_ot.append(X_i_aligned)
    return np.mean(cos_sims)
