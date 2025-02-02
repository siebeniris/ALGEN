import random
import sys
import torch
import numpy as np

seed = 42
rng= random.Random(seed)
torch.manual_seed(seed)


# (Shetty et al., 2024, WET)
def is_full_rank_circulant(first_row):
    # Compute the eigenvalues using FFT
    eigenvalues = np.fft.fft(first_row)
    # Check if all eigenvalues are non-zero
    return np.all(np.abs(eigenvalues) > 1e-10)


def get_transformation_row(n, k):
    # n: original embeddings, k: original dimensions used in transformation.
    # intialize row
    positions = rng.sample(range(n), k=k)
    # positions = self.rng.sample(range(self.args.gpt_emb_dim), k=self.args.random_transformation_dims)
    # print(f"Positions: {positions}")
    # positions.
    row = [0.0] * n
    for position in positions:
        row[position] = rng.random()
    return torch.FloatTensor(row).reshape(n)


def get_transformation_matrix_cyclic(w, n, k):
    # w: watermarked embeddings dimensions
    # logger.info("Using circulant transformation matrix")
    mat = []

    first_row = get_transformation_row(n, k)

    if not is_full_rank_circulant(first_row):
        sys.exit(1)  # TODO: for now breaking the pipeline, add retries

    curr_row = torch.clone(first_row)
    for i in range(w):
        values = torch.clone(curr_row)
        values /= torch.sum(values)  # normalise
        mat.append(values)
        curr_row = torch.roll(curr_row, 1)  # shift one right, roll the row by 1 to the right.
        if curr_row.equal(first_row):
            print(f"Row repeating, at {i + 1}")
            first_row = get_transformation_row(n, k)
            if not is_full_rank_circulant(first_row):
                sys.exit(1)  # TODO: for now breaking the pipeline, add retries
            curr_row = torch.clone(first_row)
    return torch.stack(mat)

def defense_WET(X):
    nr_sample, X_dim = X.shape
    print(f"nr sample {nr_sample}, source shape {X_dim}")

    n = X_dim
    k = n
    w = n

    # get the T
    print("calculating T for WET")
    T_trans = get_transformation_matrix_cyclic(n, k, w)
    transformation_matrix_cond = np.linalg.cond(T_trans)
    transformation_matrix_rank = np.linalg.matrix_rank(T_trans)
    print(f"transformation matrix condition {transformation_matrix_cond} and rank {transformation_matrix_rank}")
    T_trans = torch.Tensor(T_trans).to(X.device)

    X_trans_stack = []
    for i in range(nr_sample):
        X_i = X[i]
        X_i_trans = torch.mm(T_trans, X_i.reshape(X_dim, 1)).reshape(-1)
        X_i_trans_normed = X_i_trans / torch.norm(X_i_trans, p=2, dim=0, keepdim=True)
        assert len(X_i_trans_normed) == n
        assert torch.norm(X_i_trans_normed, p=2, dim=0, keepdim=True) > .999999
        X_trans_stack.append(X_i_trans_normed)

    X_trans = torch.stack(X_trans_stack, axis=0)
    print(f"shape of transformed X {X_trans.shape}")
    return X_trans


