import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss, CosineSimilarity


def visualize_losses():
    # Create sample vectors
    v1 = torch.tensor([1.0, 0.0])  # Reference vector
    angles = torch.linspace(0, 2 * np.pi, 100)

    mse_losses = []
    cos_losses = []

    mse_loss_fn = MSELoss()
    cos_sim = CosineSimilarity(dim=0)

    # Calculate losses for vectors at different angles
    for angle in angles:
        # Create rotated vector with same magnitude
        v2 = torch.tensor([torch.cos(angle), torch.sin(angle)])

        # Calculate losses
        mse = mse_loss_fn(v1, v2).item()
        cos = -cos_sim(v1, v2).item()  # Negative because we minimize this

        mse_losses.append(mse)
        cos_losses.append(cos)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(angles.numpy() * 180 / np.pi, mse_losses, label='MSE Loss')
    plt.plot(angles.numpy() * 180 / np.pi, cos_losses, label='Cosine Loss')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Loss')
    plt.title('Loss vs Angle')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    angles_subset = angles[::10]  # Plot fewer vectors for clarity
    for angle in angles_subset:
        v = torch.tensor([torch.cos(angle), torch.sin(angle)])
        plt.arrow(0, 0, v[0].item(), v[1].item(),
                  head_width=0.05, head_length=0.1, fc='blue', alpha=0.2)
    plt.arrow(0, 0, v1[0].item(), v1[1].item(),
              head_width=0.05, head_length=0.1, fc='red', label='Reference')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Vector Space')
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def demonstrate_loss_behavior():
    # Create example embeddings
    original = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Case 1: Same direction, different magnitude
    scaled = original * 2

    # Case 2: Similar direction, small perturbation
    perturbed = original + torch.randn_like(original) * 0.1

    # Case 3: Different direction
    rotated = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # 90-degree rotation

    mse_loss_fn = MSELoss()
    cos_sim = CosineSimilarity(dim=1)

    cases = {
        "Scaled (2x)": scaled,
        "Perturbed": perturbed,
        "Rotated": rotated
    }

    print("Loss behaviors for different transformations:")
    print("-" * 50)
    for name, transformed in cases.items():
        mse = mse_loss_fn(original, transformed).item()
        cos = -torch.mean(cos_sim(original, transformed)).item()
        print(f"\n{name}:")
        print(f"MSE Loss: {mse:.4f}")
        print(f"Cosine Loss: {cos:.4f}")

    return original, cases


# Run demonstrations
print("Visualizing losses across different vector transformations...")
visualize_losses()

print("\nDemonstrating specific loss behaviors...")
original, cases = demonstrate_loss_behavior()