import torch
import ot
import torch.nn as nn


def compute_transport_plan(source_embeddings, target_embeddings, source_weights, target_weights, reg=0.1):
    """
    Compute the Optimal Transport plan.
    Args:
        source_embeddings (torch.Tensor): Source embeddings [seq_length, hidden_size].
        target_embeddings (torch.Tensor): Target embeddings [seq_length, hidden_size].
        source_weights (torch.Tensor): Source weights [seq_length].
        target_weights (torch.Tensor): Target weights [seq_length].
        reg (float): Regularization parameter for Sinkhorn.

    Returns:
        torch.Tensor: Transport plan of shape [seq_length, seq_length].
    """
    # Compute the cost matrix (e.g., squared Euclidean distance)
    cost_matrix = torch.cdist(source_embeddings, target_embeddings, p=2) ** 2  # Shape: [seq_length, seq_length]

    # Convert to NumPy for OT computation
    cost_matrix_np = cost_matrix.detach().cpu().numpy()
    source_weights_np = source_weights.detach().cpu().numpy()
    target_weights_np = target_weights.detach().cpu().numpy()

    # Compute Sinkhorn transport plan
    ### TODO: incorporate ot unbalanced_sinkhorn.
    transport_plan = ot.sinkhorn(source_weights_np, target_weights_np, cost_matrix_np, reg)

    return torch.tensor(transport_plan, dtype=torch.float32, device=source_embeddings.device)


def alignment_loss_ot(source_embeddings, target_embeddings,transport_plan):
    """
        Compute the alignment loss.
        Args:
            source_embeddings (torch.Tensor): Source embeddings [seq_length, hidden_size].
            target_embeddings (torch.Tensor): Target embeddings [seq_length, hidden_size].
            transport_plan (torch.Tensor): Transport plan [seq_length, seq_length].

        Returns:
            torch.Tensor: Alignment loss.
        """
    # Align source embeddings using the transport plan
    aligned_embeddings = torch.matmul(transport_plan, target_embeddings)  # Shape: [seq_length, hidden_size]

    # Compute loss (e.g., Mean Squared Error)
    loss_fn = nn.MSELoss()
    loss = loss_fn(source_embeddings, aligned_embeddings)
    return loss

