import numpy as np
from collections import defaultdict

class EMWordAligner:
    def __init__(self, max_iterations=10, convergence_threshold=1e-6):
        self.translation_prob = None
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def initialize_probabilities(self, parallel_corpus):
        """
        Initialize translation probabilities uniformly.
        Args:
            parallel_corpus (list of tuples): List of (source_sentence, target_sentence).
        """
        self.translation_prob = defaultdict(lambda: defaultdict(float))
        vocab_source = set()
        vocab_target = set()

        # Collect vocabulary
        for source_sentence, target_sentence in parallel_corpus:
            vocab_source.update(source_sentence)
            vocab_target.update(target_sentence)

        # Uniform initialization
        uniform_prob = 1 / len(vocab_target)
        for source_word in vocab_source:
            for target_word in vocab_target:
                self.translation_prob[source_word][target_word] = uniform_prob

    def e_step(self, parallel_corpus):
        """
        Perform the expectation step: Compute alignment probabilities.
        Args:
            parallel_corpus (list of tuples): List of (source_sentence, target_sentence).

        Returns:
            count (dict): Expected count for each (source_word, target_word) pair.
            total (dict): Total counts for each source_word.
        """
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)

        for source_sentence, target_sentence in parallel_corpus:
            for source_word in source_sentence:
                normalization_factor = 0.0

                # Compute normalization factor
                for target_word in target_sentence:
                    normalization_factor += self.translation_prob[source_word][target_word]

                # Update count and total for each (source_word, target_word) pair
                for target_word in target_sentence:
                    prob = self.translation_prob[source_word][target_word] / normalization_factor
                    count[source_word][target_word] += prob
                    total[source_word] += prob

        return count, total

    def m_step(self, count, total):
        """
        Perform the maximization step: Update translation probabilities.
        Args:
            count (dict): Expected counts from the E-step.
            total (dict): Total counts from the E-step.
        """
        for source_word in count:
            for target_word in count[source_word]:
                self.translation_prob[source_word][target_word] = count[source_word][target_word] / total[source_word]

    def fit(self, parallel_corpus):
        """
        Train the EM-based word alignment model.
        Args:
            parallel_corpus (list of tuples): List of (source_sentence, target_sentence).
        """
        # Step 1: Initialize probabilities
        self.initialize_probabilities(parallel_corpus)

        # Step 2: Iterate E-step and M-step
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}")
            # E-step
            count, total = self.e_step(parallel_corpus)

            # M-step
            self.m_step(count, total)

            # Check convergence (sum of changes in probabilities)
            delta = sum(
                abs(count[source][target] - self.translation_prob[source][target])
                for source in count
                for target in count[source]
            )
            print(f"Delta: {delta}")
            if delta < self.convergence_threshold:
                print("Converged!")
                break

    def align(self, source_sentence, target_sentence):
        """
        Perform hard alignment based on learned probabilities.
        Args:
            source_sentence (list): Source sentence tokens.
            target_sentence (list): Target sentence tokens.

        Returns:
            list of tuples: Aligned word pairs (source_index, target_index).
        """
        alignments = []
        for i, source_word in enumerate(source_sentence):
            best_alignment = None
            best_prob = 0
            for j, target_word in enumerate(target_sentence):
                prob = self.translation_prob[source_word][target_word]
                if prob > best_prob:
                    best_alignment = (i, j)
                    best_prob = prob
            if best_alignment:
                alignments.append(best_alignment)
        return alignments



# Example parallel corpus
parallel_corpus = [
    (["the", "house"], ["la", "maison"]),
    (["the", "book"], ["le", "livre"]),
    (["a", "house"], ["une", "maison"]),
]

# Train the aligner
aligner = EMWordAligner(max_iterations=10)
aligner.fit(parallel_corpus)

# Align new sentences
source_sentence = ["the", "house"]
target_sentence = ["la", "maison"]
alignments = aligner.align(source_sentence, target_sentence)
print("Alignments:", alignments)

# Parallel Text Alignment: When you have a parallel corpus
# (e.g., English-French) and need to learn token alignments.
