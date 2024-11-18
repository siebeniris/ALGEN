import torch
import torch.nn as nn
import numpy as np


def create_test_orthogonal_matrix(input_dim: int, output_dim: int) -> torch.Tensor:
    """
    Create a test orthogonal matrix for testing the aligner.
    If output_dim > input_dim, pads with zeros to maintain semi-orthogonality.
    If input_dim > output_dim, takes the first output_dim rows.
    """
    # Create a square matrix of the larger dimension
    max_dim = max(input_dim, output_dim)

    # Create random matrix
    random_matrix = torch.randn(max_dim, max_dim)

    # Make it orthogonal using QR decomposition
    Q, R = torch.linalg.qr(random_matrix)

    # Make diagonal elements positive for uniqueness
    d = torch.diag(R)
    ph = d / torch.abs(d)
    Q = Q * ph

    # Resize to desired dimensions
    if output_dim > input_dim:
        # Pad with zeros
        result = torch.zeros(output_dim, input_dim)
        result[:input_dim, :] = Q[:input_dim, :input_dim]
    else:
        # Take subset
        result = Q[:output_dim, :input_dim]

    return result


def verify_orthogonality(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Verify if a matrix is orthogonal (or semi-orthogonal)
    Returns True if the matrix is orthogonal within tolerance
    """
    rows, cols = matrix.shape
    min_dim = min(rows, cols)

    if rows >= cols:
        # Tall matrix: M^T M should be identity
        product = torch.mm(matrix.t(), matrix)
        target = torch.eye(cols)
    else:
        # Wide matrix: M M^T should be identity
        product = torch.mm(matrix, matrix.t())
        target = torch.eye(rows)

    error = torch.max(torch.abs(product - target))
    return error < tol


def test_aligner():
    """
    Test the EmbeddingAlignerOrthogonal with various scenarios
    """
    # Test case 1: Square matrix (4x4)
    input_dim1, output_dim1 = 4, 4
    test_matrix1 = create_test_orthogonal_matrix(input_dim1, output_dim1)
    print(f"\nTest Case 1 - Square Matrix ({input_dim1}x{input_dim1}):")
    print("Matrix:")
    print(test_matrix1)
    print(f"Is orthogonal: {verify_orthogonality(test_matrix1)}")
    print("WW^T:")
    print(torch.mm(test_matrix1, test_matrix1.t()))

    # Test case 2: Tall matrix (6x4)
    input_dim2, output_dim2 = 4, 6
    test_matrix2 = create_test_orthogonal_matrix(input_dim2, output_dim2)
    print(f"\nTest Case 2 - Tall Matrix ({output_dim2}x{input_dim2}):")
    print("Matrix:")
    print(test_matrix2)
    print(f"Is semi-orthogonal: {verify_orthogonality(test_matrix2)}")
    print("W^TW:")
    print(torch.mm(test_matrix2.t(), test_matrix2))

    # Test case 3: Wide matrix (4x6)
    input_dim3, output_dim3 = 6, 4
    test_matrix3 = create_test_orthogonal_matrix(input_dim3, output_dim3)
    print(f"\nTest Case 3 - Wide Matrix ({output_dim3}x{input_dim3}):")
    print("Matrix:")
    print(test_matrix3)
    print(f"Is semi-orthogonal: {verify_orthogonality(test_matrix3)}")
    print("WW^T:")
    print(torch.mm(test_matrix3, test_matrix3.t()))

    return test_matrix1, test_matrix2, test_matrix3


def test_aligner_with_sequences():
    """
    Test the EmbeddingAlignerOrthogonal with sequence inputs
    """
    # Create test data
    batch_size = 2
    seq_len = 3
    input_dim = 4
    output_dim = 6

    # Create input sequence
    x = torch.tensor([
        # Batch 1
        [[1.0, 0.0, 0.0, 0.0],  # First token
         [0.0, 1.0, 0.0, 0.0],  # Second token
         [0.0, 0.0, 1.0, 0.0]],  # Third token
        # Batch 2
        [[0.0, 0.0, 0.0, 1.0],
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.5, 0.5, 0.0]]
    ], dtype=torch.float32)

    # Create orthogonal matrix
    W = create_test_orthogonal_matrix(input_dim, output_dim)

    print("\nTest Case 4 - Sequence Input:")
    print("Input shape:", x.shape)
    print("Input sequence (first batch):")
    print(x[0])
    print("\nOrthogonal matrix:")
    print(W)

    # Manual transformation
    transformed = torch.einsum('bsh,oh->bso', x, W)
    print("\nTransformed sequence shape:", transformed.shape)
    print("Transformed sequence (first batch):")
    print(transformed[0])

    return x, W, transformed


if __name__ == "__main__":
    # Run test cases
    print("Creating and testing orthogonal matrices...")
    test_matrix1, test_matrix2, test_matrix3 = test_aligner()

    print("\nTesting with sequence inputs...")
    test_x, test_W, test_transformed = test_aligner_with_sequences()

    # Test with actual aligner
    from alignment_models import EmbeddingAlignerOrthogonal

    print("\nTesting EmbeddingAlignerOrthogonal...")
    # Test case dimensions
    input_dim, output_dim = 4, 6

    # Initialize aligner
    aligner = EmbeddingAlignerOrthogonal(input_dim, output_dim, orthogonal=True)

    # Set weights to our test matrix
    with torch.no_grad():
        aligner.W.copy_(test_W)

    # Test forward pass
    output = aligner(test_x)

    print("Aligner output shape:", output.shape)
    print("Max difference between manual and aligner output:",
          torch.max(torch.abs(output - test_transformed)).item())
    print("Orthogonality error:", aligner.check_orthogonality())