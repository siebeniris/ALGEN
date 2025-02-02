import torch
import numpy as np


def insert_gaussian_noise(X, noise_level: float):
    # insert gaussian noise.
    X += noise_level * torch.randn(X.shape).to(X.device)
    # normalize along the hidden dimension.
    X = X / torch.norm(X, p=2, dim=1, keepdim=True)
    return X


def dp_guassian_embeddings(X, epsilon=1.0, delta=1e-5, sensitivity=2):
    """
    Apply the Gaussian mechanism to
    :param X: embeddings
    :param epsilon: the privacy budget.
    :param delta: the probability of exceeding epsilon.
    :param sensitivity: the sensitivity of the embedding.
    :return:
    """
    # calculate the noise scale.
    sigma = (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
    # Generate Gaussian noise
    noise = sigma * torch.normal(0, sigma, X.shape)
    # Add noise to the embeddings
    noisy_embeddings = X + noise.to(X.device)

    # normalizing the noisy embeddings
    noisy_embeddings = noisy_embeddings/ torch.norm(noisy_embeddings, p=2, dim=1, keepdim=True)

    return noisy_embeddings

