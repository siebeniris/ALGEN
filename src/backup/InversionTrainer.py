import torch
import torch.nn as nn
import torch.optim as optim

from alignment_models import (LinearAligner,
                              NeuralAligner,
                              optimal_transport_align,
                              procrustes_alignment)

from InversionModel import EmbeddingInverter
from losses import compute_transport_plan, alignment_loss_ot
from utils import get_weights_from_attention_mask
from torch.utils.data import DataLoader


# Trainer Class
class EmbeddingInverterTrainer:
    def __init__(self,
                 model: EmbeddingInverter,
                 train_loader: DataLoader,
                 eval_loader:DataLoader, # test data.
                 loss_fn, # TODO: type?
                 learning_rate: float = 3e-4,
                 weight_decay: float = 0.01):
        """
        Initialize the trainer.
        Args:
            model: EmbeddingInverter model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = model.device

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.optimizer = torch.optim.AdamW(
            # only train the aligner parameters.
            model.aligner.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.loss_fn = loss_fn

    def compute_loss_cosine(self, source_embeddings, target_embeddings,
                            attention_mask=None):
        """
        Compute the loss using cosine similarity without normalizing embeddings.
        Preserves the original magnitudes of the embeddings.
        """
        aligned_embeddings = self.model.aligner(source_embeddings)

        if attention_mask is not None:
            # Apply attention mask
            mask = attention_mask.unsqueeze(-1)
            aligned_embeddings = aligned_embeddings * mask
            target_embeddings = target_embeddings * mask

        # Compute dot product
        dot_product = torch.sum(aligned_embeddings * target_embeddings, dim=-1)  # [batch_size, seq_len]

        if attention_mask is not None:
            # Apply mask to similarities and compute mean only over valid tokens
            dot_product = dot_product * attention_mask
            loss = -torch.sum(dot_product) / torch.sum(attention_mask)
        else:
            loss = -torch.mean(dot_product)

        return loss, aligned_embeddings

    def compute_loss(self, source_embeddings, target_embeddings,
                     attention_mask=None):
        """Compute the loss between aligned source and target embeddings."""
        aligned_embeddings = self.model.aligner(source_embeddings)

        if attention_mask is not None:
            # Apply attention mask to both embeddings
            aligned_embeddings = aligned_embeddings * attention_mask.unsqueeze(-1)
            target_embeddings = target_embeddings * attention_mask.unsqueeze(-1)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(aligned_embeddings, target_embeddings)
        return loss, aligned_embeddings

    def train_step(self, text_batch, loss_func):
        """

        :param text_batch: A batch of text (a list of texts?)
        :param loss_func:
        :return:
        """
        """Perform a single training step."""
        # in the model, the model is already in the device.

        print(f"model is loaded on device {self.device}")
        self.model.train()
        self.optimizer.zero_grad()

        # Get source embeddings
        with torch.no_grad():
            source_embeddings, source_mask = self.model.get_embeddings_S(text_batch)
            target_embeddings, target_mask = self.model.get_embeddings_G(text_batch)

        if loss_func=="ot":
            source_weights = get_weights_from_attention_mask(source_mask)
            target_weights = get_weights_from_attention_mask(target_mask)

        # Compute loss
        else:
            loss, _ = self.compute_loss(source_embeddings, target_embeddings, source_mask)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, text):
        """Evaluate the model on validation data."""
        self.model.eval()
        with torch.no_grad():
            # Get embeddings
            source_embeddings, source_mask = self.model.get_embeddings_S(text)
            target_embeddings, target_mask = self.model.get_embeddings_G(text)

            # Compute loss
            loss, aligned_embeddings = self.compute_loss(
                source_embeddings,
                target_embeddings,
                source_mask
            )

            # Decode aligned embeddings
            decoded_text = self.model.decode_embeddings(aligned_embeddings, source_mask)

            # add evaluation for evaluating the decoded text and original text.
            # TODO:


        return {
            'loss': loss.item(),
            'decoded_text': decoded_text
        }

    def train(self,
              train_data,
              val_data=None,
              num_epochs=10,
              batch_size=32,
              eval_steps=100):
        """
        Train the model.
        Args:
            train_data: List of (source_text, target_text) pairs
            val_data: Optional validation data
            num_epochs: Number of training epochs
            batch_size: Batch size
            eval_steps: Evaluate every N steps
        """
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            # Create batches
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                source_texts, target_texts = zip(*batch)

                # Train step
                loss = self.train_step(list(source_texts), list(target_texts))
                total_loss += loss
                num_batches += 1

                # Evaluate periodically
                if num_batches % eval_steps == 0 and val_data is not None:
                    self.evaluate_and_log(val_data, epoch, num_batches)

            # Log epoch results
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def evaluate_and_log(self, val_data, epoch, batch):
        """Helper function to evaluate and log results."""
        eval_source, eval_target = zip(*val_data[:5])  # Take first 5 examples
        eval_results = self.evaluate(list(eval_source), list(eval_target))

        print(f"\nEvaluation at Epoch {epoch + 1}, Batch {batch}:")
        print(f"Validation Loss: {eval_results['loss']:.4f}")
        print("Sample Reconstructions:")
        for src, tgt, dec in zip(eval_source, eval_target, eval_results['decoded_text']):
            print(f"\nSource: {src}")
            print(f"Target: {tgt}")
            print(f"Decoded: {dec}")
            print("-" * 50)
