from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class TokenAligner:
    def __init__(self, src_tokenizer, tgt_tokenizer):
        """
        Initialize the TokenAligner with source and target tokenizers.

        Args:
            src_tokenizer: Source tokenizer (e.g., BERT tokenizer)
            tgt_tokenizer: Target tokenizer (e.g., GPT tokenizer)
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def get_token_alignments(self,
                             text: str,
                             src_embeddings: torch.Tensor,
                             tgt_embeddings: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Get token-level alignments between source and target embeddings.

        Args:
            text: Input text that was encoded by both tokenizers
            src_embeddings: Token embeddings from source model (B x S x D)
            tgt_embeddings: Token embeddings from target model (B x T x D)

        Returns:
            List of tuples containing aligned token indices (src_idx, tgt_idx)
        """
        # Get tokenizations
        src_tokens = self.src_tokenizer.tokenize(text)
        tgt_tokens = self.tgt_tokenizer.tokenize(text)

        # Calculate token-level similarity matrix
        sim_matrix = F.cosine_similarity(
            src_embeddings.unsqueeze(2),
            tgt_embeddings.unsqueeze(1),
            dim=-1
        )

        # Get alignments using optimal transport
        alignments = self._optimal_transport_matching(
            sim_matrix,
            src_tokens,
            tgt_tokens
        )

        return alignments

    def align_embeddings(self,
                         src_emb: torch.Tensor,
                         tgt_emb: torch.Tensor,
                         sentence_alignment_matrix: torch.Tensor) -> torch.Tensor:
        """
        Align token embeddings using sentence-level alignment as guidance.

        Args:
            src_emb: Source token embeddings (B x S x D)
            tgt_emb: Target token embeddings (B x T x D)
            sentence_alignment_matrix: Alignment matrix between sentences (B x B)

        Returns:
            Aligned source embeddings in target space
        """
        # Project source embeddings to target space using linear transformation
        projection = nn.Linear(src_emb.size(-1), tgt_emb.size(-1))
        projected_src = projection(src_emb)

        # Use sentence alignments as soft attention weights
        attention_weights = F.softmax(sentence_alignment_matrix, dim=-1)

        # Apply attention-weighted averaging
        aligned_embeddings = torch.bmm(
            attention_weights.unsqueeze(1),
            projected_src
        )

        return aligned_embeddings

    def _optimal_transport_matching(self,
                                    sim_matrix: torch.Tensor,
                                    src_tokens: List[str],
                                    tgt_tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Find optimal token alignments using Sinkhorn-Knopp algorithm.

        Args:
            sim_matrix: Token similarity matrix
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens

        Returns:
            List of aligned token indices
        """
        # Convert similarity to cost matrix (higher similarity = lower cost)
        cost_matrix = 1 - sim_matrix

        # Run Sinkhorn-Knopp algorithm
        n_iters = 50
        epsilon = 0.1

        log_a = torch.zeros(len(src_tokens))
        log_b = torch.zeros(len(tgt_tokens))

        for _ in range(n_iters):
            log_a = epsilon * (
                    torch.log(torch.ones(len(src_tokens))) -
                    torch.logsumexp((-cost_matrix + log_b.unsqueeze(0)) / epsilon, dim=1)
            )
            log_b = epsilon * (
                    torch.log(torch.ones(len(tgt_tokens))) -
                    torch.logsumexp((-cost_matrix + log_a.unsqueeze(1)) / epsilon, dim=0)
            )

        transport_matrix = torch.exp(
            (-cost_matrix + log_a.unsqueeze(1) + log_b.unsqueeze(0)) / epsilon
        )

        # Extract alignments from transport matrix
        alignments = []
        while transport_matrix.max() > 0.1:  # Threshold for considering alignment
            i, j = divmod(transport_matrix.argmax().item(), transport_matrix.size(1))
            alignments.append((i, j))
            transport_matrix[i, :] = 0
            transport_matrix[:, j] = 0

        return alignments


# Example usage:
"""
# Initialize tokenizers
src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tgt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize models and get embeddings
text = "Example sentence for alignment."
src_embeddings = ... # Get embeddings from source model
tgt_embeddings = ... # Get embeddings from target model
sentence_alignment = ... # Get sentence-level alignment matrix

# Create aligner and align embeddings
aligner = TokenAligner(src_tokenizer, tgt_tokenizer)
token_alignments = aligner.get_token_alignments(text, src_embeddings, tgt_embeddings)
aligned_embeddings = aligner.align_embeddings(
    src_embeddings, 
    tgt_embeddings, 
    sentence_alignment
)
"""