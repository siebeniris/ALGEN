import torch
import torch.nn as nn
from utils import sinkhorn


class AlignerOT(nn.Module):
    def __init__(self, source_dimension, target_dimension, device, scale: float = 300):
        super(AlignerOT, self).__init__()
        # Load pre-trained embeddings
        # Dimensions
        self.device = device
        self.source_dimension = source_dimension
        self.target_dimension = target_dimension
        print(f"initializing source {self.source_dimension} and target {self.target_dimension}")

        # Transformation matrix for aligning source embeddings to target embedings
        # after their dimension already aligned
        self.delta_ot = nn.Parameter(torch.FloatTensor(self.target_dimension, self.target_dimension),
                                     requires_grad=True)

        nn.init.xavier_uniform_(self.delta_ot)

        # Scaling factor for OT cost
        self.scale = scale

    def cal_ot(self, source_embeddings, target_embeddings, delta_ot):
        """
        Parallelized version for calculating Optimal Transport (OT) matrix.
        """
        device = delta_ot.device
        batch_size, seq_length, source_dim = source_embeddings.shape
        target_dim = target_embeddings.shape[-1]

        # Uniform distributions for OT, each element has the same probabilities
        source_dis = torch.full((seq_length, source_dim), 1 / source_dim, device=device)
        target_dis = torch.full((seq_length, target_dim), 1 / target_dim, device=device)

        # Compute cost matrices for all sequences in the batch
        cost = ((source_embeddings.unsqueeze(3) - target_embeddings.unsqueeze(2)) ** 2) * self.scale
        # cost shape: (batch_size, seq_length, source_dim, target_dim)

        # Initialize list to store future tasks
        futures = []

        # Launch parallel Sinkhorn tasks
        for b in range(batch_size):
            futures.append(
                torch.jit.fork(
                    lambda b_idx: torch.stack(
                        [sinkhorn(source_dis[t], target_dis[t], cost[b_idx, t])[0] for t in range(seq_length)]
                    ),
                    b  # Pass the current batch index
                )
            )

        # Collect results from parallel tasks
        sinkhorn_results = [torch.jit.wait(f) for f in futures]
        sinkhorn_results = torch.stack(sinkhorn_results)  # (batch_size, seq_length, source_dim, target_dim)

        # Return averaged OT matrix adjusted with delta_ot
        return sinkhorn_results.mean(dim=(0, 1)) * target_dim * self.scale + delta_ot  # (source_dim, target_dim)

    def forward(self, X, Y):
        """
        Aligns ling_vec to img_vec using Optimal Transport.
        """
        device = self.device

        source_vec = torch.tensor(X).to(torch.float32)  # source embeddings
        target_vec = torch.tensor(Y).to(torch.float32)  # target embeddings

        # Compute OT matrix to align ling_vec to img_vec
        ot_matrix = self.cal_ot(source_vec.to(device), target_vec.to(device), delta_ot=self.delta_ot)

        # Align ling_vec to img_vec
        aligned_ling_vec = source_vec.to(device).mm(ot_matrix.to(device))

        return aligned_ling_vec
