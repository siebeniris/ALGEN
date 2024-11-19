import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re
import Levenshtein
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm, trange
from transformers.modeling_outputs import BaseModelOutput

from utils import get_weights_from_attention_mask
import ot


class LinearAligner(nn.Module):
    def __init__(self, source_dim, target_dim):
        # align from source space to target space
        super().__init__()
        self.W = nn.Linear(source_dim, target_dim, bias=False)

    def forward(self, x):
        return self.W(x)


class NeuralAligner(nn.Module):
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        print(f"Initializing NeuralAligner with source_dim={source_dim}, target_dim={target_dim}")

        hidden_dim = source_dim
        self.network = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x):
        original_shape = x.shape
        print("original shape", original_shape)

        output = self.network(x)
        return output


class EmbeddingAlignerOrthogonal(nn.Module):
    """
    A neural network that learns to align two embedding spaces.
    Handles sequence inputs of shape (batch_size, seq_len, hidden_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, orthogonal: bool = True):
        """
        Args:
            input_dim (int): Dimension of source embedding space
            output_dim (int): Dimension of target embedding space
            orthogonal (bool): Whether to enforce orthogonality constraint
        """
        super().__init__()
        self.orthogonal = orthogonal
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the linear transformation matrix
        self.W = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using random orthogonal matrix"""
        nn.init.orthogonal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform source embeddings to target space
        Args:
            x (torch.Tensor): Source embeddings of shape (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Transformed embeddings of shape (batch_size, seq_len, output_dim)
        """
        # check the dimension.
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, hidden_dim), got shape {x.shape}")

        batch_size, seq_len, hidden_dim = x.shape

        if hidden_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {hidden_dim}")

        if self.orthogonal:
            # Apply orthogonality constraint using SVD
            try:
                u, s, v = torch.linalg.svd(self.W, full_matrices=False)
                W = torch.mm(u, v)  # v is already transposed in torch.linalg.svd
            except RuntimeError:
                W = self.W
                print("Warning: SVD failed, using non-orthogonal weights")
        else:
            W = self.W

        # Method 1: Reshape and batch process
        # Reshape to (batch_size * seq_len, hidden_dim)
        x_reshaped = x.view(-1, hidden_dim)
        # Apply transformation
        transformed = torch.mm(x_reshaped, W.t())
        # Reshape back to (batch_size, seq_len, output_dim)
        output = transformed.view(batch_size, seq_len, self.output_dim)

        # Alternative Method 2: Using einsum (more readable but might be slower)
        # output = torch.einsum('bsh,oh->bso', x, W)

        return output

    @torch.no_grad()
    def check_orthogonality(self) -> float:
        """
        Check how close W is to being orthogonal.
        Returns:
            float: Mean squared error from identity matrix
        """
        if self.W.shape[0] != self.W.shape[1]:
            return torch.tensor(float('inf'))  # Non-square matrices aren't orthogonal

        WWT = torch.mm(self.W, self.W.t())
        I = torch.eye(self.W.shape[0], device=self.W.device)
        error = torch.mean((WWT - I) ** 2).item()
        return error


def optimal_transport_align(source_embeddings, target_embeddings,
                            source_attention_mask, target_attention_mask,
                            reg=0.1, reg_m=10.0):
    """
    Align source embeddings to target embeddings using Optimal Transport with cosine distance normalized to [0, 1].

    Args:
        source_embeddings (torch.Tensor): Embeddings from source space [batch_size, seq_length, hidden_size].
        target_embeddings (torch.Tensor): Embeddings from target space [batch_size, seq_length, hidden_size].
        source_attention_mask (torch.Tensor):  [batch_size, seq_length]
        target_attention_mask (torch.Tensor):  [batch_size, seq_length]
        reg (float): Entropy regularization.
        reg (float): Regularization parameter for entropy regularized OT.
        reg_m (float): Marginal constraint relaxation regularization.

    Returns:
        aligned_embeddings (torch.Tensor): Source embeddings aligned to the target space.
    """
    # Ensure embeddings are on the same device
    source_embeddings = source_embeddings.cpu().numpy()  # [batch_size, seq_length, hidden_size]
    target_embeddings = target_embeddings.cpu().numpy()  # [batch_size, seq_length, hidden_size]

    same_tokenizer = False
    # check first if the attention masks are close, if not, then they are not from the same tokenizers for sure
    if source_attention_mask.shape == target_attention_mask.shape \
            and torch.equal(source_attention_mask, target_attention_mask):
        same_tokenizer = True

    print("same tokenizer:", same_tokenizer)

    # source_weights (np.array): Weights for source distribution [seq_length_source].
    # target_weights (np.array): Weights for target distribution [seq_length_target].
    source_weights = get_weights_from_attention_mask(source_attention_mask)
    target_weights = get_weights_from_attention_mask(target_attention_mask)

    # check if the weights are balanced, then decide if we use OT.balanced or not.
    are_balanced = False
    if source_weights.shape == target_weights.shape:
        are_balanced = torch.allclose(source_weights, target_weights, atol=1e-6)
    print("balanced weights:", are_balanced)

    # move weights to numpy
    source_weights = source_weights.cpu().numpy()
    target_weights = target_weights.cpu().numpy()

    batch_size, seq_length, hidden_size = source_embeddings.shape
    print(f"batch size: {batch_size}, seq length: {seq_length}, hidden size: {hidden_size}")

    # when the tokenizers are the same, the weights are the same.
    # then we do not need OT balanced or unbalanced to do alignment.
    # in this case, we align the embeddings using cost matrix (Distance between embeddings)
    if are_balanced:
        print("running balanced OT alignemnt")
        aligned_embeddings = []
        for b in range(batch_size):
            source_embeddings_batch = source_embeddings[b]
            target_embeddings_batch = target_embeddings[b]

            source_weights_batch = source_weights[b]
            target_weights_batch = target_weights[b]
            # seq_length x seq_length.
            # source embeddings (13, 768) (seq_length x hidden_size)
            cost_matrix = 1 - ((cosine_similarity(source_embeddings_batch, target_embeddings_batch) + 1) / 2)
            transport_plan = ot.sinkhorn(source_weights_batch, target_weights_batch, cost_matrix, reg=reg)
            # Align source embeddings to the target space
            aligned_embedding = np.dot(transport_plan, target_embeddings_batch)
            aligned_embeddings.append(aligned_embedding)

        aligned_embeddings = np.array(aligned_embeddings)
        return aligned_embeddings

    else:
        print(f"Running unbalanced OT")
        aligned_embeddings = []
        for b in range(batch_size):
            # normalized
            source_embeddings_batch = source_embeddings[b]
            target_embeddings_batch = target_embeddings[b]
            source_weights_batch = source_weights[b]
            target_weights_batch = target_weights[b]
            cost_matrix = 1 - ((cosine_similarity(source_embeddings_batch, target_embeddings_batch) + 1) / 2)
            # sinkhorn_unbalanced, doesn't need source and target embeddings to be
            transport_plan = ot.sinkhorn_unbalanced(source_weights_batch, target_weights_batch, cost_matrix, reg, reg_m)
            # print(f"transport plan: {transport_plan.shape}")
            aligned_embedding = np.dot(transport_plan, target_embeddings_batch)
            aligned_embeddings.append(aligned_embedding)
            # print(f"aligned embedding: {aligned_embedding.shape}")

        aligned_embeddings = np.array(aligned_embeddings)
        return aligned_embeddings


def procrustes_alignment(source_embeddings, target_embeddings):
    """
    Align source embeddings to target embeddings using Procrustes analysis
    for embeddings with different hidden dimensions.

    Args:
        source_embeddings (np.array): Source embeddings of shape [N, D_S].
        target_embeddings (np.array): Target embeddings of shape [N, D_G].

    Returns:
        aligned_embeddings (np.array): Source embeddings aligned to the target space.
        W (np.array): Transformation matrix of shape [D_S, D_G].
    """
    # Center the embeddings
    source_mean = source_embeddings.mean(axis=0)  # [D_S]
    target_mean = target_embeddings.mean(axis=0)  # [D_G]
    source_centered = source_embeddings - source_mean  # [N, D_S]
    target_centered = target_embeddings - target_mean  # [N, D_G]

    # Compute the cross-covariance matrix
    covariance_matrix = np.dot(source_centered.T, target_centered)  # [D_S, D_G]

    # Perform SVD on the covariance matrix
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the transformation matrix W
    W = np.dot(U, Vt)  # [D_S, D_G]

    # Align the source embeddings S to the target space G
    aligned_embeddings = np.dot(source_centered, W)  # [N, D_G]

    return aligned_embeddings, W


if __name__ == '__main__':
    # Example dimensions
    source_dim = 768  # Your actual source embedding dimension
    target_dim = 512  # Your actual target embedding dimension

    # Create model
    model = NeuralAligner(source_dim, target_dim)

    # Test with sample input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, source_dim)

    # Forward pass
    try:
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")