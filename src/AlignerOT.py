import ot
import numpy as np
import torch
import torch.nn as nn
from utils import pairwise_cosine, sinkhorn


class AlignerOT(nn.Module):
    def __init__(self, source_dimension, target_dimension, device, scale: float = 300):
        super(AlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        print(f"initializing source {self.source_dimension} and target {self.target_dimension}")

        # Transformation matrix for aligning source emebddings to target emebddings
        self.delta_ot = nn.Parameter(torch.FloatTensor(self.target_dimension,
                                self.target_dimension),
                                requires_grad=True)
        self.mats_align = nn.Linear(self.source_dimension, self.target_dimension)
        self.mats_align = self.mats_align.to(device)
        nn.init.xavier_uniform_(self.delta_ot)

        # Scaling factor for OT cost
        self.scale = scale

    def cal_ot(self, source_embeddings, target_embeddings, delta_ot):
        """
        Calculates the Optimal Transport (OT) matrix to align source_embeddings to target_embeddings.
        """
        device = delta_ot.device

        number = source_embeddings.shape[0]  # Number of samples for averaging
        source_dim = source_embeddings.shape[-1]
        target_dim = target_embeddings.shape[-1]

        # Uniform distributions for OT, each element has the same probabilities
        source_dis = torch.ones_like(source_embeddings[0, :]) / source_embeddings.shape[-1]
        target_dis = torch.ones_like(target_embeddings[0, :]) / target_embeddings.shape[-1]

        # print(source_dis.shape, target_dis.shape)

        # Initialize OT matrices
        matrix_temp = torch.zeros((number, source_dim, target_dim), device=device)

        # Compute Sinkhorn distance over multiple samples
        # sinkhorn: solve OT efficiently, use entropy regularization to make OT more stable and scalable.
        # min_T <T,C> - \epsilon H(T) (entropy of the transportation plan)
        with torch.no_grad():
            # for each token: there is a transport plan.
            for i in range(number):
                cost = ((source_embeddings[i, :].unsqueeze(0) - target_embeddings[i, :].unsqueeze(1)) ** 2) * self.scale
                matrix_temp[i, :, :] = sinkhorn(source_dis, target_dis, cost)[0]  # [number,sourece_dim,target_dim]

        # Return averaged OT matrix adjusted with delta_ot
        return matrix_temp.mean(dim=0) * target_dim * self.scale + delta_ot  # [sourece_dim, target_dim]

    def forward(self, X, Y):
        """
        Aligns ling_vec to img_vec using Optimal Transport.
        """
        device = self.device
        source_vec = torch.tensor(X).to(torch.float32)  # source embeddings
        target_vec = torch.tensor(Y).to(torch.float32)  # target embeddings
        # align source to target vectors
        source_vec = self.mats_align(source_vec)  # (seq_length, target_dim)
        # Compute OT matrix to align ling_vec to img_vec
        ot_matrix = self.cal_ot(source_vec.to(device), target_vec.to(device), delta_ot=self.delta_ot)

        # Align ling_vec to img_vec
        aligned_ling_vec = source_vec.to(device).mm(ot_matrix.to(device))

        return aligned_ling_vec


class SeqTokenAlignerOT(nn.Module):
    def __init__(self, source_dim, target_dim, source_seq_len, target_seq_len, device):
        super(SeqTokenAlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_dim = source_dim
        self.target_dim = target_dim

        self.linear = nn.Linear(self.source_dim, self.target_dim)

        print(f"initializing source {self.source_dim}  and target {self.target_dim}")
        # Transformation matrix for aligning source embeddings to target embeddings
        self.delta_ot = nn.Parameter(torch.FloatTensor(
                                    source_seq_len,
                                    target_seq_len),
                                    requires_grad=True)
        nn.init.xavier_uniform_(self.delta_ot)
        # nn.init.uniform_(self.delta_ot, -0.001, 0.001)
        self.scale = nn.Parameter(torch.ones(1))
        # Scaling factor for OT cost

    def optimal_transport_weight(self, sim, reg=0.1, reg_m=0.001, ot_strategy="ub_sinkhorn"):
        """
        Calculates the Optimal Transport (OT) matrix to align source_embeddings to target_embeddings.
        """
        # TODO: need to deal with this in batch

        # n x m token dimension
        M = -sim.data.cpu().numpy()
        n, m = M.shape
        a = np.ones(n) / n
        b = np.ones(m) / m
        if ot_strategy == "ub_sinkhorn":
            weight = torch.FloatTensor(ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg, reg_m=reg_m))
        else:
            weight = torch.FloatTensor(ot.emd(a, b, M))

        weight = weight.to(self.device)
        return weight

    def forward(self, X, Y):
        """
        Aligns source to target embeddings using Optimal Transport.
        """
        # align source to target vectors
        # seq_len x seq_len
        X_aligned = self.linear(X)
        A = pairwise_cosine(X_aligned, Y)
        W = self.optimal_transport_weight(A, ot_strategy="ub_sinkhorn")
        # seq_len x seq_len ,
        T = A * W + self.delta_ot
        X_aligned = torch.mm(T, X_aligned)
        return X_aligned


class TokenAlignerOT(nn.Module):
    def __init__(self, source_token_len, target_token_len, device):
        super(TokenAlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_token_len = source_token_len
        self.target_token_len = target_token_len

        print(f"initializing source {self.source_token_len} and target {self.target_token_len}")
        # Transformation matrix for aligning source embeddings to target embeddings
        self.delta_ot = nn.Parameter(torch.FloatTensor(
                                    self.source_token_len,
                                    self.target_token_len),
                                    requires_grad=True)
        nn.init.xavier_uniform_(self.delta_ot)
        # nn.init.uniform_(self.delta_ot, -0.001, 0.001)
        # self.scale = nn.Parameter(torch.ones(1))
        # Scaling factor for OT cost

    def optimal_transport_weight(self, sim, reg=0.1, reg_m=0.001, ot_strategy="ub_sinkhorn"):
        """
        Calculates the Optimal Transport (OT) matrix to align source_embeddings to target_embeddings.
        """
        # TODO: need to deal with this in batch

        # n x m token dimension
        M = -sim.data.cpu().numpy()
        n, m = M.shape
        a = np.ones(n) / n
        b = np.ones(m) / m
        if ot_strategy == "ub_sinkhorn":
            weight = torch.FloatTensor(ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg, reg_m=reg_m))
        else:
            weight = torch.FloatTensor(ot.emd(a, b, M))

        weight = weight.to(self.device)
        return weight

    def forward(self, X, Y):
        """
        Aligns source to target embeddings using Optimal Transport.
        """
        # align source to target vectors
        # seq_len x seq_len
        A = pairwise_cosine(X, Y)
        W = self.optimal_transport_weight(A, ot_strategy="ub_sinkhorn")
        # seq_len x seq_len
        T = A * W  + self.delta_ot
        X_aligned = torch.mm(T, X)
        return X_aligned


class SingleLinear(nn.Module):
    def __init__(self, emb1_dim, emb2_dim):
        super(SingleLinear, self).__init__()
        self.linear = nn.Linear(emb1_dim, emb2_dim)

    def forward(self, emb1):
        return self.linear(emb1)


class OtSimLoss(nn.Module):
    def __init__(self, device="cuda", eps=1e-6, reg=0.1, reg_m=0.01):
        super(OtSimLoss, self).__init__()
        self.eps = eps
        self.device = device
        self.reg = reg
        self.reg_m = reg_m

    def optimal_transport_weight(self, sim):
        M = - sim.data.cpu().numpy()
        num_u, num_p = M.shape

        a = np.ones(num_u)/num_u
        b = np.ones(num_p)/num_p

        weight = torch.FloatTensor(ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=self.reg, reg_m=self.reg_m))

        weight = weight.to(self.device)
        return weight

    def optimal_transport(self, sim):
        weight = self.optimal_transport_weight(sim)
        return torch.sum(weight * sim)

    def forward(self, emb1, emb2):
        if not emb1.requires_grad:
            emb1.requires_grad_(True)
        if not emb2.requires_grad:
            emb2.requires_grad_(True)
        A = pairwise_cosine(emb1, emb2, self.eps)
        ot_loss = self.optimal_transport_weight(A)
        return torch.relu(1-ot_loss)
