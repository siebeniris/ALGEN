import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class RecursiveDecoder:
    def __init__(
            self,
            model_name: str = "t5-base",
            max_length: int = 128,
            num_recursive_steps: int = 3,
            sequence_beam_width: int = 5,
            decoding_strategy: str = "beam",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.max_length = max_length
        self.num_recursive_steps = num_recursive_steps
        self.sequence_beam_width = sequence_beam_width
        self.decoding_strategy = decoding_strategy

        # Initialize model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Generation parameters
        self.return_best_hypothesis = True  # Use embedding similarity for selection

    def decode_embeddings(
            self,
            embedding_g: torch.Tensor,  # Teacher embedding
            aligned_embedding_s: torch.Tensor,  # Aligned student embedding
            attention_mask: torch.Tensor = None
    ) -> list:
        """
        Decode embeddings using recursive refinement.

        Args:
            embedding_g: Teacher model embeddings
            aligned_embedding_s: Aligned student embeddings
            attention_mask: Optional attention mask

        Returns:
            list: Generated text sequences
        """
        batch_size = embedding_g.size(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                embedding_g.size(1),
                dtype=torch.long,
                device=embedding_g.device
            )

        # Initial generation inputs
        inputs = {
            "embedding_g": embedding_g,
            "aligned_embedding_s": aligned_embedding_s,
            "attention_mask": attention_mask
        }

        num_steps_remaining = self.num_recursive_steps
        steps_completed = 0
        best_scores_seen = None

        # Prepare generation kwargs based on strategy
        generation_kwargs = self._get_generation_kwargs()

        current_hypothesis = None
        current_hypothesis_embedding = aligned_embedding_s

        while num_steps_remaining > 0:
            # Generate new hypothesis and get embeddings
            hypothesis_ids, hypothesis_embedding, scores = self._generate_step(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                current_hypothesis=current_hypothesis,
                current_hypothesis_embedding=current_hypothesis_embedding
            )

            # Update for next iteration
            current_hypothesis = hypothesis_ids
            current_hypothesis_embedding = hypothesis_embedding

            # Early stopping check
            if scores is not None:
                if best_scores_seen is not None and torch.isclose(
                        scores, best_scores_seen, atol=1e-3
                ).all():
                    print(f"Scores converged after {steps_completed} steps")
                    break
                best_scores_seen = scores

            num_steps_remaining -= 1
            steps_completed += 1

        # Decode final hypothesis
        return self.tokenizer.batch_decode(
            current_hypothesis,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def _generate_step(
            self,
            inputs: dict,
            generation_kwargs: dict,
            current_hypothesis: torch.Tensor = None,
            current_hypothesis_embedding: torch.Tensor = None
    ) -> tuple:
        """Single step of recursive generation."""
        batch_size = inputs["embedding_g"].size(0)

        # Create encoder outputs
        encoder_outputs = BaseModelOutput(
            last_hidden_state=current_hypothesis_embedding or inputs["aligned_embedding_s"]
        )

        # Initialize decoder input IDs
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        # Generate
        outputs = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs
        )

        generated_ids = outputs.sequences

        # Compute sequence scores
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True
        )
        sequence_scores = transition_scores.sum(axis=1)

        # Re-embed generated text
        generated_embedding = self._embed_text(generated_ids)

        # Select best hypotheses
        if generated_ids.size(0) > batch_size:
            beam_width = int(generated_ids.size(0) / batch_size)

            # Compute similarities
            similarities = torch.nn.CosineSimilarity(dim=2)(
                generated_embedding.reshape(batch_size, beam_width, -1),
                inputs["embedding_g"].unsqueeze(1)
            )

            # Select based on similarity or generation scores
            scores = similarities if self.return_best_hypothesis else sequence_scores.reshape(batch_size, beam_width)
            best_indices = scores.argmax(dim=1)

            # Gather best hypotheses
            generated_ids = generated_ids.reshape(batch_size, beam_width, -1)[
                torch.arange(batch_size), best_indices
            ]
            generated_embedding = generated_embedding.reshape(batch_size, beam_width, -1)[
                torch.arange(batch_size), best_indices
            ]

            best_scores = scores.max(dim=1).values
        else:
            best_scores = None

        return generated_ids, generated_embedding, best_scores

    def _get_generation_kwargs(self) -> dict:
        """Get generation parameters based on decoding strategy."""
        kwargs = {
            "max_length": self.max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.decoding_strategy == "beam":
            kwargs.update({
                "num_beams": max(self.sequence_beam_width, 5),
                "num_return_sequences": max(self.sequence_beam_width, 5),
                "length_penalty": 1.0,
                "early_stopping": True,
                "do_sample": False
            })
        elif self.decoding_strategy == "nucleus":
            kwargs.update({
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7
            })

        return kwargs

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Re-embed generated text."""
        with torch.no_grad():
            outputs = self.model.encoder(input_ids=input_ids)
            return outputs.last_hidden_state


# Example usage:
"""
decoder = RecursiveDecoder(
    model_name="t5-base",
    num_recursive_steps=3,
    sequence_beam_width=5
)

# Assuming you have embedding_g and aligned_embedding_s
generated_text = decoder.decode_embeddings(
    embedding_g=embedding_g,
    aligned_embedding_s=aligned_embedding_s
)
"""