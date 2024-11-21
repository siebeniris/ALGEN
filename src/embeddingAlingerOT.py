import torch
import torch.nn as nn
from utils import sinkhorn


class AlignerOT(nn.Module):
    def __init__(self, source_dimension, target_dimension, device, scale: float = 300):
        super(AlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        print(f"initializing source {self.source_dimension} and target {self.target_dimension}")

        # Transformation matrix for aligning source emebddings to target emebddings
        self.delta_ot = nn.Parameter(torch.Tensor(self.target_dimension, self.target_dimension), requires_grad=True)
        self.mats_align = nn.Linear(self.source_dimension, self.target_dimension)
        self.delta_ot = self.delta_ot.to(device)
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

        # Uniform distributions for OT
        source_dis = torch.ones_like(source_embeddings[0, :]) / source_embeddings.shape[-1]
        target_dis = torch.ones_like(target_embeddings[0, :]) / target_embeddings.shape[-1]

        # print(source_dis.shape, target_dis.shape)

        # Initialize OT matrices
        matrix_temp = torch.zeros((number, source_dim, target_dim), device=device)

        # Compute Sinkhorn distance over multiple samples
        with torch.no_grad():
            for i in range(number):
                cost = ((source_embeddings[i, :].unsqueeze(0) - target_embeddings[i, :].unsqueeze(1)) ** 2) * self.scale
                matrix_temp[i, :, :] = sinkhorn(source_dis, target_dis, cost)[0]

        # Return averaged OT matrix adjusted with delta_ot
        return matrix_temp.mean(dim=0) * target_dim * self.scale + delta_ot

    def forward(self, X, Y):
        """
        Aligns ling_vec to img_vec using Optimal Transport.
        """
        device = self.delta_ot.device

        source_vec = torch.tensor(X).to(torch.float32)  # source embeddings
        target_vec = torch.tensor(Y).to(torch.float32)  # target embeddings

        # align source to target vectors
        source_vec = self.mats_align(source_vec) # (seq_length, target_dim)

        # Compute OT matrix to align ling_vec to img_vec
        ot_matrix = self.cal_ot(source_vec.to(device), target_vec.to(device), delta_ot=self.delta_ot)

        # Align ling_vec to img_vec
        aligned_ling_vec = source_vec.to(device).mm(ot_matrix.to(device))

        return aligned_ling_vec
