import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot


class EmbeddingAlignerOT(nn.Module):
    def __init__(self,
                 s_hidden_size, g_hidden_size,
                 adjust_weights_with_magnitutde=True,
                 ot_reg=0.1, ot_reg_m=10.0):
        """
        Initialize the embedding aligner with linear transformation and optimal transport.

        Args:
            source_dim (int): Dimension of source embeddings
            target_dim (int): Dimension of target embeddings
            ot_reg (float): Entropy regularization parameter for optimal transport
            ot_reg_m (float): Marginal relaxation parameter for unbalanced optimal transport
        """
        super().__init__()

        self.s_hidden_size = s_hidden_size
        self.g_hidden_size = g_hidden_size

        self.adjust_weights_with_magnitutde = adjust_weights_with_magnitutde

        self.ot_reg = ot_reg
        self.ot_reg_m = ot_reg_m

        # Linear transformation layer
        self.linear_transform = nn.Linear(self.s_hidden_size, self.g_hidden_size,
                                          bias=True)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.zeros_(self.linear_transform.bias)

    def get_adjusted_weights_with_magnitude(self, attention_mask, embeddings, scale_factor=1.0):
        """
        Adjust weights using attention mask and embedding magnitude.

        Args:
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length].
            embeddings (torch.Tensor): Token embeddings [batch_size, seq_length, hidden_dim].
            scale_factor (float): Scaling factor for magnitudes (optional).

        Returns:
            torch.Tensor: Adjusted and normalized weights [batch_size, seq_length].
        """
        # Compute the magnitude of each token embedding
        eps = 1e-12
        magnitudes = torch.norm(embeddings, p=2, dim=-1) + eps  # [batch_size, seq_length]

        # Combine attention mask with magnitudes
        adjusted_weights = attention_mask.float() * (magnitudes ** scale_factor) + eps

        # Normalize weights to sum to 1 across the sequence dimension
        adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        return adjusted_weights

    def get_weights_from_mask(self, attention_mask):
        """
        Convert attention mask to distribution weights.

        Args:
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length]

        Returns:
            torch.Tensor: Normalized weights [batch_size, seq_length]
        """
        weights = attention_mask.float()
        # Normalize weights to sum to 1 for each sequence in batch
        # dim=-1 in this case is dim=1
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return weights

    def compute_weighted_euclidean_cost_matrix(self, s_embeddings, g_embeddings, s_weights, g_weights):
        """
        Compute weighted Euclidean distance cost matrix between source and target embeddings.

        Args:
            s_embeddings (torch.Tensor): Source embeddings [seq_length_S, hidden_dim]
            g_embeddings (torch.Tensor): Target embeddings [seq_length_G, hidden_dim]
            s_weights (torch.Tensor): Source weights [seq_length_S]
            g_weights (torch.Tensor): Target weights [seq_length_G]

        Returns:
            torch.Tensor: Weighted cost matrix [seq_length_S, seq_length_G]
        """
        # Get dimensions
        seq_length_S, hidden_dim = s_embeddings.shape
        seq_length_G, _ = g_embeddings.shape

        # Reshape embeddings for pairwise distance computation
        s_expanded = s_embeddings.unsqueeze(1)  # [seq_length_S, 1, hidden_dim]
        g_expanded = g_embeddings.unsqueeze(0)  # [1, seq_length_G, hidden_dim]

        # Reshape sequence weights for broadcasting
        s_weights = s_weights.view(seq_length_S, 1)  # [seq_length_S, 1]
        g_weights = g_weights.view(1, seq_length_G)  # [1, seq_length_G]

        # Compute pairwise distances
        # [seq_length_S, seq_length_G, hidden_dim]
        squared_diff = (s_expanded - g_expanded) ** 2

        # Sum along hidden dimension to get unweighted distances
        # [seq_length_S, seq_length_G]
        distances = torch.sqrt(torch.sum(squared_diff, dim=-1))

        # Apply sequence weights through broadcasting
        # [seq_length_S, seq_length_G]
        weighted_distances = distances * s_weights * g_weights

        return weighted_distances

    def optimal_transport_align(self, transformed_s_embeddings, g_embeddings,
                                s_attention_mask, g_attention_mask, scale=1):
        """
        Align embeddings using optimal transport.

        Args:
            transformed_s_embeddings (torch.Tensor): Linear transformed source embeddings
            g_embeddings (torch.Tensor): Target embeddings
            s_weights (torch.Tensor): weights
            g_weights (torch.Tensor): weights from g_attention_mask

        Returns:
            torch.Tensor: OT-aligned embeddings
        """

        batch_size = transformed_s_embeddings.size(0)

        # for all the batches .
        if self.adjust_weights_with_magnitutde:
            s_weights = self.get_adjusted_weights_with_magnitude(s_attention_mask, transformed_s_embeddings, scale)
            g_weights = self.get_adjusted_weights_with_magnitude(g_attention_mask, g_embeddings, scale)
        else:
            s_weights = self.get_weights_from_mask(s_attention_mask)
            g_weights = self.get_weights_from_mask(g_attention_mask)

            # check if the weights are balanced, then decide if we use OT.balanced or not.
        are_balanced = False
        if s_weights.shape == g_weights.shape:
            are_balanced = torch.allclose(s_weights, g_weights, atol=1e-8)

        print("balanced weights:", are_balanced)

        aligned_embeddings = []
        for b in range(batch_size):
            # Get sequence embeddings and masks for current batch
            # tensors
            source_emb_batch = transformed_s_embeddings[b]
            target_emb_batch = g_embeddings[b]
            source_weights_batch = s_weights[b]
            target_weights_batch = g_weights[b]

            # numpy array for cost_matrix
            source_weights_np = source_weights_batch.detach().cpu().numpy()
            target_weights_np = target_weights_batch.detach().cpu().numpy()

            cost_matrix = self.compute_weighted_euclidean_cost_matrix(source_emb_batch, target_emb_batch,
                                                                      source_weights_batch,
                                                                      target_weights_batch).detach().cpu().numpy()
            cost_matrix += 1e-12
            if are_balanced:
                ot_plan = ot.sinkhorn(
                    source_weights_batch, target_weights_batch,
                    cost_matrix, reg=self.ot_reg,
                    numItermax=100,
                    stopThr=1e-8

                )

            else:
                ot_plan = ot.sinkhorn_unbalanced(
                    source_weights_np, target_weights_np,
                    cost_matrix, reg=self.ot_reg, reg_m=self.ot_reg_m,
                    numItermax=100,
                    stopThr=1e-8,
                )
            # convert transport plan to tensor
            ot_plan = torch.tensor(
                ot_plan, device=transformed_s_embeddings.device,
                dtype=transformed_s_embeddings.dtype
            )

            # In optimal_transport_align:
            print(f"Source weights range: [{source_weights_np.min():.6f}, {source_weights_np.max():.6f}]")
            print(f"Target weights range: [{target_weights_np.min():.6f}, {target_weights_np.max():.6f}]")
            print(f"Cost matrix range: [{cost_matrix.min():.6f}, {cost_matrix.max():.6f}]")
            print(f"OT plan range: [{ot_plan.min():.6f}, {ot_plan.max():.6f}]")

            aligned_embed = torch.mm(ot_plan, target_emb_batch)
            aligned_embeddings.append(aligned_embed)

        return torch.stack(aligned_embeddings)

    def forward(self, source_embeddings, target_embeddings,
                source_attention_mask, target_attention_mask, scale=1):
        """
        Forward pass of the aligner.

        Args:
            source_embeddings (torch.Tensor): Source embeddings [batch_size, seq_length, hidden_size]
            target_embeddings (torch.Tensor): Target embeddings [batch_size, seq_length, hidden_size]
            source_attention_mask (torch.Tensor): Source attention mask [batch_size, seq_length]
            target_attention_mask (torch.Tensor): Target attention mask [batch_size, seq_length]

        Returns:
            torch.Tensor: Aligned embeddings in target space
        """
        # Apply linear transformation
        transformed_embeddings = self.linear_transform(source_embeddings)

        # Apply optimal transport alignment
        aligned_embeddings = self.optimal_transport_align(
            transformed_embeddings, target_embeddings,
            source_attention_mask, target_attention_mask, scale=scale
        )

        return aligned_embeddings


if __name__ == '__main__':
    def test_embedding_aligner():
        # Create synthetic data
        batch_size = 3
        seq_len_s = 10  # Source sequence length
        seq_len_g = 10  # Target sequence length
        hidden_size_s = 128  # Source embedding dimension
        hidden_size_g = 256  # Target embedding dimension

        # Generate random embeddings
        s_embeddings = torch.rand(batch_size, seq_len_s, hidden_size_s)
        g_embeddings = torch.rand(batch_size, seq_len_g, hidden_size_g)

        # Generate random attention masks (binary: 1 for valid, 0 for padding)
        s_attention_mask = torch.randint(0, 2, (batch_size, seq_len_s))
        g_attention_mask = torch.randint(0, 2, (batch_size, seq_len_g))

        # Initialize EmbeddingAligner
        aligner = EmbeddingAlignerOT(
            s_hidden_size=hidden_size_s,
            g_hidden_size=hidden_size_g,
            adjust_weights_with_magnitutde=True,
            ot_reg=0.1,
            ot_reg_m=10.0
        )

        # Perform alignment
        aligned_embeddings = aligner(
            source_embeddings=s_embeddings,  # Assuming input embeddings are transformed
            target_embeddings=g_embeddings,
            source_attention_mask=s_attention_mask, target_attention_mask=g_attention_mask,
            scale=1
        )

        # Assertions and print results
        assert aligned_embeddings.shape == g_embeddings.shape, \
            f"Expected aligned_embeddings shape {g_embeddings.shape}, got {aligned_embeddings.shape}"
        print("Aligned embeddings shape:", aligned_embeddings.shape)
        print("Aligned embeddings (example):", aligned_embeddings[0])


    # Run the test
    test_embedding_aligner()
