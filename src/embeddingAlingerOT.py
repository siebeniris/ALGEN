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

        # Transformation matrix for aligning source emebddings to target emebddings
        self.delta_ot = nn.Parameter(torch.FloatTensor(self.target_dimension, self.target_dimension),
                                     requires_grad=True)
        # self.mats_align = nn.Linear(self.source_dimension, self.target_dimension)
        # self.mats_align = self.mats_align.to(device)
        nn.init.xavier_uniform_(self.delta_ot)

        # Scaling factor for OT cost
        self.scale = scale

    def cal_ot(self, source_embeddings, target_embeddings, delta_ot):
        """
        Calculates the Optimal Transport (OT) matrix to align source_embeddings to target_embeddings.
        """
        device = delta_ot.device
        # TODO: need to deal with this in batch
        number = source_embeddings.shape[0]  # Number of samples for averaging

        source_dim = source_embeddings.shape[-1]
        target_dim = target_embeddings.shape[-1]

        # Uniform distributions for OT, each element has the same probabilities
        #  [dim]/dim.shape[-1]
        source_dis = torch.ones_like(source_embeddings[0, :]) / source_embeddings.shape[-1]
        target_dis = torch.ones_like(target_embeddings[0, :]) / target_embeddings.shape[-1]

        ###########################################################
        # Initialize OT matrices
        # matrix_temp = torch.zeros((number, source_dim, target_dim), device=device)
        # # Compute Sinkhorn distance over multiple samples
        # # sinkhorn: solve OT efficiently, use entropy regularization to make OT more stable and scalable.
        # # min_T <T,C> - \epsilon H(T) (entropy of the transportation plan)
        # with torch.no_grad():
        #     # for each token: there is a transport plan.
        #     for i in range(number):
        #         # source_embeddings[i, :].unsqueeze(0)  [1,dim]
        #         cost = ((source_embeddings[i, :].unsqueeze(0) - target_embeddings[i, :].unsqueeze(1)) ** 2) * self.scale
        #         matrix_temp[i, :, :] = sinkhorn(source_dis, target_dis, cost)[0]  # [number,sourece_dim,target_dim]
        #######################################################################

        cost = ((source_embeddings.unsqueeze(2) - target_embeddings.unsqueeze(1)) ** 2) * self.scale

        with torch.no_grad():
            matrix_temp = sinkhorn(source_dis, target_dis, cost)[0]
            # matrix_temp shape: (batch_size, source_dim, target_dim)

        # Return averaged OT matrix adjusted with delta_ot
        return matrix_temp.mean(dim=0) * target_dim * self.scale + delta_ot  # [sourece_dim, target_dim]

    def forward(self, X, Y):
        """
        Aligns ling_vec to img_vec using Optimal Transport.
        """
        device = self.device

        # source_vec = torch.tensor(X).to(torch.float32)  # source embeddings
        # target_vec = torch.tensor(Y).to(torch.float32)  # target embeddings

        # align source to target vectors
        source_vec = X  # (seq_length, target_dim)
        target_vec = Y
        batch_size, seq_len, hidden_dim = Y.shape
        source_vec = source_vec.view(-1, source_vec.shape[-1])
        target_vec = target_vec.view(-1, target_vec.shape[-1])

        # Compute OT matrix to align ling_vec to img_vec
        ot_matrix = self.cal_ot(source_vec.to(device), target_vec.to(device), delta_ot=self.delta_ot)
        print(source_vec.shape, ot_matrix.shape)
        # Align ling_vec to img_vec
        # aligned_ling_vec = torch.bmm(source_vec.to(device), ot_matrix.to(device))

        aligned_ling_vec = source_vec.to(device).mm(ot_matrix.to(device))
        aligned_ling_vec = aligned_ling_vec.view(batch_size, seq_len, hidden_dim)
        return aligned_ling_vec
