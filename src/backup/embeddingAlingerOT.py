import torch
import torch.nn as nn
from utils import sinkhorn


class AlignerOT(nn.Module):
    def __init__(self, source_dimension, target_dimension, device, scale: float=500):
        super(AlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        self.scale = scale

        print(f"initializing source {self.source_dimension} and target {self.target_dimension}")
        # Transformation matrix for aligning source emebddings to target emebddings
        self.delta_ot = nn.Parameter(torch.FloatTensor(self.target_dimension, self.target_dimension),
                                     requires_grad=True)

        # Initialize delta_ot closer to identity
        # self.delta_ot = nn.Parameter(
        #     torch.eye(self.target_dimension) + 0.01 * torch.randn(self.target_dimension, self.target_dimension),
        #     requires_grad=True
        # )
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
        # TODO: need to deal with this in batch

        batch_size, seq_len, hidden_dim = source_embeddings.shape

        # Compute Sinkhorn distances for all sequences in the batch
        source_vec_stacks = []
        for i in range(batch_size):
            with torch.no_grad():
                matrix_temp = torch.zeros((seq_len, hidden_dim, hidden_dim), device=device)
                for t in range(seq_len):
                    cost = ((source_embeddings[i, t, :].unsqueeze(0) - target_embeddings[i, t, :].unsqueeze(
                        1)) ** 2) * self.scale  # [dim,dim]
                    # Uniform 1D distributions for each token [1, dim]
                    source_dis = torch.ones_like(source_embeddings[i, t, :], device=device) / hidden_dim
                    target_dis = torch.ones_like(target_embeddings[i, t, :], device=device) / hidden_dim
                    # print(f'cost in seq {cost.shape}, target dis {target_dis.shape}, source {source_dis.shape}')
                    # source_dis [dim,] target_dis [dim,] would transposition make a difference?
                    p = sinkhorn(source_dis, target_dis, cost)[0]
                    # print(p)
                    matrix_temp[t, :, :] = p
                    # source_embeddings_aligned = source_embeddings[i][t].mm(p)
                    # matrix_temp[i, :, :] shape: (hidden_dim, hidden_dim)
                tplan = matrix_temp.mean(dim=0).to(device)  # dim on batch.
                # tplan = tplan / (tplan.sum() + 1e-8) # add normalization
            tplan = tplan * hidden_dim * self.scale + delta_ot  # (hidden_dim, hidden_dim)
            aligned_source_vec_batch = source_embeddings[i].mm(tplan)
            source_vec_stacks.append(aligned_source_vec_batch)

        # Average the OT matrices over the sequence length and batch dimensions
        # averaged_ot_matrix = matrix_temp.mean(dim=(0, 1))  # Shape: (hidden_dim, hidden_dim)
        return torch.stack(source_vec_stacks)

    def forward(self, X, Y):
        """
        Aligns source to target embeddings using Optimal Transport.
        """
        device = self.device
        # align source to target vectors
        source_vec = X  # (seq_length, target_dim)
        target_vec = Y

        if X.shape[-1] != Y.shape[-1]:
            source_vec = self.mats_align(source_vec)

        # Compute OT matrix to align ling_vec to img_vec
        aligned_source_vec = self.cal_ot(source_vec.to(device), target_vec.to(device), delta_ot=self.delta_ot)
        return aligned_source_vec
