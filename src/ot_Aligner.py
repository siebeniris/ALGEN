import torch
import torch.nn as nn
import ot

class OTAligner(nn.Module):
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.W = nn.Linear(source_dim, target_dim, bias=False)  # Learnable transformation matrix

    def forward(self, source_embeddings):
        return self.W(source_embeddings)


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
    transport_plan = ot.sinkhorn(source_weights_np, target_weights_np, cost_matrix_np, reg)

    return torch.tensor(transport_plan, dtype=torch.float32, device=source_embeddings.device)


def alignment_loss(source_embeddings, target_embeddings, transport_plan):
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


def train_ot_aligner(model, source_embeddings_list, target_embeddings_list, source_weights_list, target_weights_list, epochs=100, lr=0.01, reg=0.1):
    """
    Train the OT-based aligner model.
    Args:
        model (nn.Module): OTAligner model.
        source_embeddings_list (list): List of source embeddings tensors [batch_size, seq_length, source_dim].
        target_embeddings_list (list): List of target embeddings tensors [batch_size, seq_length, target_dim].
        source_weights_list (list): List of source weights tensors [batch_size, seq_length].
        target_weights_list (list): List of target weights tensors [batch_size, seq_length].
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        reg (float): Sinkhorn regularization parameter.

    Returns:
        nn.Module: Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for source_embeddings, target_embeddings, source_weights, target_weights in zip(
            source_embeddings_list, target_embeddings_list, source_weights_list, target_weights_list
        ):
            # Move tensors to the same device as the model
            source_embeddings = source_embeddings.to(next(model.parameters()).device)
            target_embeddings = target_embeddings.to(next(model.parameters()).device)
            source_weights = source_weights.to(next(model.parameters()).device)
            target_weights = target_weights.to(next(model.parameters()).device)

            # Forward pass: Transform source embeddings
            transformed_embeddings = model(source_embeddings)

            # Compute transport plan
            transport_plan = compute_transport_plan(transformed_embeddings, target_embeddings, source_weights, target_weights, reg)

            # Compute alignment loss
            loss = alignment_loss(transformed_embeddings, target_embeddings, transport_plan)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

    return model

# Example data
batch_size = 3
seq_length = 13
source_dim = 768
target_dim = 512

source_embeddings_list = [torch.rand(seq_length, source_dim) for _ in range(batch_size)]
target_embeddings_list = [torch.rand(seq_length, target_dim) for _ in range(batch_size)]
source_weights_list = [torch.ones(seq_length) / seq_length for _ in range(batch_size)]
target_weights_list = [torch.ones(seq_length) / seq_length for _ in range(batch_size)]

# Initialize model
model = OTAligner(source_dim, target_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
trained_model = train_ot_aligner(
    model, source_embeddings_list, target_embeddings_list, source_weights_list, target_weights_list, epochs=50, lr=0.01, reg=0.1
)

# Use the trained model for alignment
source_embeddings = torch.rand(seq_length, source_dim)
aligned_embeddings = trained_model(source_embeddings)
print("Aligned Embeddings Shape:", aligned_embeddings.shape)  # Expected: [13, 512]


#
# Key Considerations
# Why Use OT?
# OT ensures that the alignment respects the structure and distribution of both source and target embeddings.
# It can handle imbalances in token distributions via Sinkhorn unbalanced OT.
# Regularization:
# reg
# reg controls the entropy of the transport plan. A higher
# reg
# reg results in smoother transport plans but might lose sharpness in alignment.
#
# Training Stability:
# Ensure cost matrices are normalized and numerical instabilities are avoided.
# Scalability:
# For large-scale datasets, compute OT in mini-batches and use parallel processing where possible.
