import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm


class EMEmbeddingMapper:
    def __init__(self, n_components=100, max_iter=100, tol=1e-6):
        """
        Initialize EM algorithm for mapping embeddings from one dimension to another.

        Args:
            n_components (int): Target number of components (100 in this case)
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covs = None
        self.weights = None

    def fit_transform(self, X):
        """
        Fit the EM algorithm and transform the input embeddings.

        Args:
            X: Input embeddings of shape (batch_size, source_dim, feature_dim)
                In this case (10, 80, 768)

        Returns:
            Transformed embeddings of shape (batch_size, target_dim, feature_dim)
                In this case (10, 100, 768)
        """
        batch_size, source_dim, feature_dim = X.shape

        # Initialize parameters for each batch
        self.means = np.zeros((batch_size, self.n_components, feature_dim))
        self.covs = np.array([np.eye(feature_dim) for _ in range(self.n_components)])
        self.covs = np.tile(self.covs[None, :, :, :], (batch_size, 1, 1, 1))
        self.weights = np.ones((batch_size, self.n_components)) / self.n_components

        # Process each batch separately
        transformed_embeddings = []
        for b in range(batch_size):
            transformed = self._fit_transform_single(X[b])
            transformed_embeddings.append(transformed)

        return np.stack(transformed_embeddings)

    def _fit_transform_single(self, X):
        """
        Apply EM algorithm to a single batch of embeddings.

        Args:
            X: Single batch of embeddings (source_dim, feature_dim)

        Returns:
            Transformed embeddings (target_dim, feature_dim)
        """
        source_dim, feature_dim = X.shape
        prev_likelihood = -np.inf

        # Initialize means using random points from input
        indices = np.random.choice(source_dim, self.n_components, replace=False)
        self.means[0] = X[indices]

        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            responsibilities = self._e_step(X)

            # M-step: Update parameters
            self._m_step(X, responsibilities)

            # Check convergence
            current_likelihood = self._compute_likelihood(X)
            if abs(current_likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = current_likelihood

        # Generate new embeddings using learned parameters
        return self._generate_embeddings()

    def _e_step(self, X):
        """
        Expectation step: Calculate responsibilities.
        """
        responsibilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            # Calculate likelihood for each component
            likelihood = multivariate_normal.pdf(
                X,
                mean=self.means[0, k],
                cov=self.covs[0, k]
            )
            responsibilities[:, k] = likelihood * self.weights[0, k]

        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """
        Maximization step: Update parameters.
        """
        # Update weights
        self.weights[0] = responsibilities.sum(axis=0) / X.shape[0]

        # Update means
        for k in range(self.n_components):
            self.means[0, k] = (responsibilities[:, k:k + 1] * X).sum(axis=0) / responsibilities[:, k].sum()

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[0, k]
            self.covs[0, k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
            # Add small diagonal term for numerical stability
            self.covs[0, k] += 1e-6 * np.eye(self.covs[0, k].shape[0])

    def _compute_likelihood(self, X):
        """
        Compute log likelihood of the data.
        """
        likelihood = 0
        for k in range(self.n_components):
            comp_likelihood = multivariate_normal.pdf(
                X,
                mean=self.means[0, k],
                cov=self.covs[0, k]
            )
            likelihood += (comp_likelihood * self.weights[0, k]).sum()
        return np.log(likelihood)

    def _generate_embeddings(self):
        """
        Generate new embeddings using learned parameters.
        """
        new_embeddings = np.zeros((self.n_components, self.means.shape[-1]))

        for k in range(self.n_components):
            new_embeddings[k] = np.random.multivariate_normal(
                self.means[0, k],
                self.covs[0, k]
            )

        return new_embeddings


# Example usage:
def map_embeddings(source_embeddings, target_shape):
    """
    Map source embeddings to target shape using EM algorithm.

    Args:
        source_embeddings: Input embeddings of shape (batch_size, source_dim, feature_dim)
        target_shape: Tuple of (batch_size, target_dim, feature_dim)

    Returns:
        Mapped embeddings of shape target_shape
    """
    mapper = EMEmbeddingMapper(n_components=target_shape[1])
    return mapper.fit_transform(source_embeddings)


# Usage example:
if __name__ == "__main__":
    # Example data
    source_embeddings = np.random.randn(10, 80, 768)  # Your input shape after first step
    target_shape = (10, 100, 768)  # Desired output shape

    # Map embeddings
    mapped_embeddings = map_embeddings(source_embeddings, target_shape)
    print(f"Mapped embeddings shape: {mapped_embeddings.shape}")