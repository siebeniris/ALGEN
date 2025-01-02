import torch
import torch.nn as nn
from utils import pairwise_cosine
import numpy as np
import ot
from tqdm import tqdm
cos = nn.CosineSimilarity(dim=1)


class EmbedAlignerOT(nn.Module):
    def __init__(self, source_dimension, target_dimension,
                 ot_strategy,
                 device,
                 reg=1, reg_m=1):
        super(EmbedAlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        self.reg = reg
        self.reg_m = reg_m
        self.ot_strategy = ot_strategy

    def optimal_transport_weight(self, sim):
        # n x m tokenization.
        # the tokens are not padded.
        M = - sim.data.cpu().numpy()
        n, m = M.shape
        a = np.ones(n) / n
        b = np.ones(m) / m
        if self.ot_strategy == "ub_sinkhorn":
            weight = torch.FloatTensor(ot.unbalanced.sinkhorn_unbalanced(a, b, M, self.reg, self.reg_m))
        else:
            weight = torch.FloatTensor(ot.emd(a, b, M))
        weight = weight.to(self.device)
        return weight

    def forward(self, source_embeddings, target_embeddings):
        A = pairwise_cosine(source_embeddings, target_embeddings)
        W = self.optimal_transport_weight(A)
        T = A * W  # (n,m)
        # print(T.shape)
        aligned_embeddings = torch.mm(T.t(), source_embeddings)
        return aligned_embeddings, T


if __name__ == '__main__':
    # Example inputs
    device = "cpu"
    # source_embeddings = torch.randn(10, 128).to(device)  # 10 source embeddings
    # target_embeddings = torch.randn(15, 128).to(device)  # 15 target embeddings
    data_path = "results/t5-base_to_google-flan-t5-small_gridsearch/X_Y_data.pth"

    data = torch.load(data_path, map_location=device)

    X_aligned = data["X_aligned"]
    X_test_aligned = data["X_test_aligned"]
    Y = data["Y"]
    Y_test = data["Y_test"]

    batch_size, seq_len, X_dim = X_aligned.shape

    # Initialize the model
    aligner = EmbedAlignerOT(
        source_dimension=128,
        target_dimension=128,
        ot_strategy="ub_sinkhorn",
        device=device,
        reg=1,
        reg_m=1
    )

    # Align embeddings

    cossim_l = []
    for i in tqdm(range(batch_size)):
        X_i = X_aligned[i]
        Y_i = Y[i]
        X_i_aligned, T = aligner(X_i, Y_i)
        cossim = cos(X_i_aligned, Y_i).mean()
        cossim_l.append(cossim.detach().cpu().numpy())

    print("X AND y", np.mean(cossim_l))


