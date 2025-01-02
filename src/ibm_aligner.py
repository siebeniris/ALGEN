import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class IBMEmbeddingAligner:
    def __init__(self, num_iterations: int = 10, max_fertility: int = 2):
        self.num_iterations = num_iterations
        self.max_fertility = max_fertility
        self.alignment_probs = None
        self.position_probs = None
        self.reordering_probs = None

    def train(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray):
        """
        Train the alignment model using source and target embeddings, considering position, fertility, and reordering.

        Args:
            source_embeddings (np.ndarray): Embeddings for the source language (shape: [num_source_words, embedding_dim]).
            target_embeddings (np.ndarray): Embeddings for the target language (shape: [num_target_words, embedding_dim]).
        """
        num_source = source_embeddings.shape[0]
        num_target = target_embeddings.shape[0]

        # Initialize alignment probabilities uniformly
        self.alignment_probs = np.full((num_source, num_target), 1 / num_target)

        # Initialize position and reordering probabilities
        self.position_probs = np.full((num_source, num_target), 1 / num_target)
        self.reordering_probs = np.full((num_source, num_target), 1 / num_target)

        # Compute cosine similarities
        cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")

            # E-step: Compute soft counts with fertility and reordering
            soft_counts = np.zeros_like(self.alignment_probs)
            for i in range(num_source):
                normalization_factor = np.sum(
                    self.alignment_probs[i] * self.position_probs[i] * self.reordering_probs[i] * cosine_sim_matrix[i]
                )
                if normalization_factor > 0:
                    soft_counts[i] = (
                        self.alignment_probs[i]
                        * self.position_probs[i]
                        * self.reordering_probs[i]
                        * cosine_sim_matrix[i]
                    ) / normalization_factor

            # M-step: Update alignment probabilities
            self.alignment_probs = soft_counts / soft_counts.sum(axis=0, keepdims=True)

            # Update position probabilities based on relative positions
            for i in range(num_source):
                for j in range(num_target):
                    self.position_probs[i, j] = np.exp(-abs(i - j))

            # Update reordering probabilities based on word jumps
            for i in range(num_source):
                for j in range(num_target):
                    self.reordering_probs[i, j] = 1.0 / (1 + abs(i - j))

    def align(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Perform alignment using trained alignment probabilities.

        Args:
            source_embeddings (np.ndarray): Embeddings for the source language (shape: [num_source_words, embedding_dim]).
            target_embeddings (np.ndarray): Embeddings for the target language (shape: [num_target_words, embedding_dim]).

        Returns:
            List[Tuple[int, int, float]]: List of alignments (source_idx, target_idx, probability).
        """
        if self.alignment_probs is None:
            raise ValueError("Model is not trained yet. Call train() first.")

        alignments = []
        cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
        for i in range(source_embeddings.shape[0]):
            for j in range(target_embeddings.shape[0]):
                alignment_prob = (
                    self.alignment_probs[i, j]
                    * self.position_probs[i, j]
                    * self.reordering_probs[i, j]
                    * cosine_sim_matrix[i, j]
                )
                alignments.append((i, j, alignment_prob))

        return sorted(alignments, key=lambda x: x[2], reverse=True)

# Example Usage
if __name__ == "__main__":
    # Example embeddings for source and target (randomly generated)
    source_embeddings = np.random.rand(5, 128)  # 5 source words, 128-dimensional embeddings
    target_embeddings = np.random.rand(7, 128)  # 7 target words, 128-dimensional embeddings

    aligner = IBMEmbeddingAligner(num_iterations=5, max_fertility=2)
    aligner.train(source_embeddings, target_embeddings)
    alignments = aligner.align(source_embeddings, target_embeddings)

    print("Alignments (source_idx, target_idx, probability):")
    for alignment in alignments:
        print(alignment)
