import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss, CosineEmbeddingLoss
import numpy as np
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from typing import List, Dict
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from InversionModel import EmbeddingInverter
from create_dataset import EmbeddingDataset, custom_collate_fn
from data_helper import load_data


class EmbeddingInverterTrainer:
    def __init__(
            self,
            model_G_name: str = "t5-base",
            model_S_name: str = "t5-small",
            save_dir: str = "checkpoints",
            checkpoint_path: str = None,
            resume_training: bool = False,
            use_wandb: bool = True,
            align_method: str = "linear",
            learning_rate: float = 1e-4,
            batch_size: int = 16,
            num_epochs: int = 100,
            max_length: int = 128,
            decoding_strategy: str = "beam",
            dataset_name: str = "flores",
            language_script: str = "eng_Latn",
            train_samples: int = 1000,
            eval_samples: int = 200,
    ):
        self.align_method = align_method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.model_G_name = model_G_name
        self.model_S_name = model_S_name
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.decoding_strategy = decoding_strategy

        self.dataset_name = dataset_name
        self.language_script = language_script
        self.train_samples = train_samples
        self.eval_samples = eval_samples

        self.model_G_name = model_G_name
        self.model_S_name = model_S_name

        # Initialize model
        self.model = EmbeddingInverter(
            model_G_name_or_path=model_G_name,
            model_S_name_or_path=model_S_name,
            max_length=max_length,
            align_method=align_method,
            decoding_strategy=decoding_strategy,
        )

        self.num_workers = 2

        # Load from checkpoint if provided
        self.start_epoch = 0
        self.best_eval_loss = float('inf')
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, resume_training)
        else:
            # Initialize optimizer if not loading from checkpoint
            self.optimizer = AdamW(self.model.aligner.parameters(), lr=learning_rate)

        # Initialize losses
        self.mse_loss = MSELoss()
        self.cos_loss = CosineEmbeddingLoss()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # todo : put all the arguments into config.
        if use_wandb:
            wandb.init(
                project=f"embedding-inverter-{self.align_method}-{self.num_epochs}",
                config={
                    "model_g": model_G_name,
                    "model_s": model_S_name,
                    'align_method': self.align_method,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs,
                    'max_length': self.max_length,
                    'model_G_name': self.model_G_name,
                    'model_S_name': self.model_S_name,
                    "save_dir": self.save_dir,
                    "checkpoint_path": self.checkpoint_path,
                    'decoding_strategy': self.decoding_strategy
                }
            )

        os.makedirs(save_dir, exist_ok=True)

    def compute_token_f1(self, pred_texts: List[str], target_texts: List[str]) -> float:
        """TODO: Change this when it is in different Languages"""
        f1_scores = []
        for pred, target in zip(pred_texts, target_texts):
            pred_tokens = set(pred.split())
            target_tokens = set(target.split())

            if len(pred_tokens) == 0 or len(target_tokens) == 0:
                continue

            intersection = pred_tokens & target_tokens
            precision = len(intersection) / len(pred_tokens)
            recall = len(intersection) / len(target_tokens)

            if precision + recall == 0:
                continue

            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def compute_metrics(self, pred_texts: List[str], target_texts: List[str]) -> Dict[str, float]:
        """Compute BLEU, ROUGE, and token F1 scores"""
        # Prepare texts for BLEU
        references = [[text.split()] for text in target_texts]
        hypotheses = [text.split() for text in pred_texts]

        # Calculate BLEU
        bleu_score = corpus_bleu(references, hypotheses)

        # Calculate ROUGE
        rouge_scores = defaultdict(list)
        for pred, target in zip(pred_texts, target_texts):
            scores = self.rouge_scorer.score(target, pred)
            for metric, score in scores.items():
                rouge_scores[f"{metric}_f"].append(score.fmeasure)

        rouge_scores = {k: np.mean(v) for k, v in rouge_scores.items()}

        # Calculate token F1
        token_f1 = self.compute_token_f1(pred_texts, target_texts)

        metrics = {
            "bleu": bleu_score,
            "token_f1": token_f1,
            **rouge_scores
        }

        return metrics

    def compute_embedding_similarity(self, aligned_embeddings: torch.Tensor, target_embeddings: torch.Tensor,
                                     attention_mask: torch.Tensor = None) -> Dict[str, float]:
        """Compute embedding space similarity metrics"""
        # Convert to numpy for sklearn cosine similarity
        aligned_np = aligned_embeddings.detach().cpu().numpy()
        target_np = target_embeddings.detach().cpu().numpy()

        if attention_mask is not None:
            mask_np = attention_mask.detach().cpu().numpy()
            # Only consider non-padded tokens
            aligned_np = aligned_np[mask_np.astype(bool)]
            target_np = target_np[mask_np.astype(bool)]

        # Compute cosine similarity
        # [0= identical, 2=opposite]
        cos_dist = np.mean([ - cosine_similarity(a.reshape(1, -1), t.reshape(1, -1))[0][0]
                            for a, t in zip(aligned_np, target_np)])

        # Compute MSE
        mse = np.mean((aligned_np - target_np) ** 2)

        return {
            "embedding_cosine_distance": cos_dist,
            "embedding_mse": mse
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        # print(batch)

        # Forward pass

        aligned_embeddings = self.model(batch)

        aligned_embeddings_reshaped = aligned_embeddings.view(-1, aligned_embeddings.size(-1))
        target_embeddings_reshaped = batch["emb_g"].view(-1, batch["emb_g"].size(-1))
        batch_seq_len = target_embeddings_reshaped.shape[0]
        target = torch.ones(batch_seq_len)

        # cosine loss
        cos_loss = self.cos_loss(
            # [batch_size*seq_len, hidden_dim]
            aligned_embeddings_reshaped, target_embeddings_reshaped, target,
            reduction="mean"
        )

        # Compute MSE loss
        mse_loss = self.mse_loss(aligned_embeddings, batch["emb_g"])

        # Weighted combination
        # TODO: we can also add weights for the losses,
        loss = mse_loss + cos_loss

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return {
            "train_loss": loss.item(),
            "train_mse_loss": mse_loss.item(),
            "train_cos_loss": cos_loss.item()
        }

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        all_metrics = defaultdict(list)
        all_texts = []
        all_decoded_texts = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Get aligned embeddings and decoded text
                aligned_embeddings = self.model(batch)
                # might need to change later when tokenizers aren't the same.
                decoded_texts = self.model.decode_embeddings(aligned_embeddings, batch["attention_mask_s"])

                # Compute embedding similarities
                emb_metrics = self.compute_embedding_similarity(
                    aligned_embeddings, batch["emb_g"], batch["attention_mask_s"]
                )

                # Store texts for later metric computation
                all_texts.extend(batch["text"])
                all_decoded_texts.extend(decoded_texts)

                # Store embedding metrics
                for k, v in emb_metrics.items():
                    all_metrics[k].append(v)

        # Compute text generation metrics
        text_metrics = self.compute_metrics(all_decoded_texts, all_texts)

        # Combine and average all metrics
        final_metrics = {}
        for k, v in all_metrics.items():
            final_metrics[f"eval_{k}"] = np.mean(v)
        for k, v in text_metrics.items():
            final_metrics[f"eval_{k}"] = v

        self.model.train()
        return final_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'metrics': metrics,
            'config': {
                'align_method': self.align_method,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'max_length': self.max_length,
                'model_G_name': self.model_G_name,
                'model_S_name': self.model_S_name,
                "save_dir": self.save_dir,
                "checkpoint_path": self.checkpoint_path,
                "use_wandb": self.use_wandb,
                'decoding_strategy': self.decoding_strategy
            }
        }

        # Save regular checkpoint
        checkpoint_dir = os.path.join(self.save_dir, f"{self.align_method}_epochs{self.num_epochs}" )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir,  f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_model_path = os.path.join(checkpoint_dir,
                                           f'best_model_{self.align_method}.pt')
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model to {best_model_path}")

        print(f"Saved checkpoint to {checkpoint_path}")

        # Remove old checkpoints to save space (keep only last 3)
        self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Remove old checkpoints, keeping only the last n"""
        checkpoints = [f for f in os.listdir(self.save_dir)
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(os.path.join(self.save_dir, checkpoint))

    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """Load model and training state from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            # Load training state
            self.optimizer = AdamW(self.model.aligner.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_eval_loss = checkpoint['best_eval_loss']

            # Log resumed training
            print(f"Resuming training from epoch {self.start_epoch}")
            print(f"Best eval loss so far: {self.best_eval_loss}")
            if self.use_wandb:
                wandb.run.summary["resumed_from_epoch"] = checkpoint['epoch']
        else:
            # Only load model weights for fine-tuning
            print("Loaded model weights for fine-tuning with fresh training state")
            self.optimizer = AdamW(self.model.aligner.parameters(), lr=self.learning_rate)

    def load_best_model(self):
        """Load the best performing model"""
        best_model_path = os.path.join(self.save_dir, f'best_model_{self.align_method}.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.model.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {best_model_path}")
            return checkpoint['metrics']
        else:
            print("No best model checkpoint found")
            return None

    def train(self):
        """Main training loop"""
        # Create datasets

        # TODO: change this , DATA IS OVERLAPING.
        all_texts = load_data(self.dataset_name, self.language_script, nr_samples=self.train_samples)
        split_index= int(len(all_texts))*0.8
        train_texts = all_texts[:split_index]
        eval_texts = all_texts[split_index:]

        train_dataset = EmbeddingDataset(train_texts, self.model)
        eval_dataset = EmbeddingDataset(eval_texts, self.model)
        print(f"num workers for dataloader {self.num_workers}")
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,  # if you're using multiple workers
            pin_memory=False,  # if you're using GPU
            drop_last=False,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            pin_memory=False,  # if you're using GPU
        )

        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_metrics = defaultdict(list)

            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in pbar:
                step_metrics = self.train_step(batch)

                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)
                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_metrics.items()})

            # Evaluation
            eval_metrics = self.evaluate(eval_dataloader)

            # Log metrics including loss
            metrics = {
                # train_losses
                **{k: np.mean(v) for k, v in epoch_metrics.items()},
                **eval_metrics
            }

            if self.use_wandb:
                wandb.log(metrics, step=epoch)

            # Save best model
            # Save checkpoint and check for best model
            is_best = eval_metrics["eval_embedding_mse"] < self.best_eval_loss
            if is_best:
                self.best_eval_loss = eval_metrics["eval_embedding_mse"]

            self.save_checkpoint(epoch, metrics, is_best=is_best)

            print(f"Epoch {epoch + 1} metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")


def main():
    # Load data
    # Initialize trainer
    trainer = EmbeddingInverterTrainer(
        align_method="linear",
        learning_rate=1e-4,
        batch_size=128,
        num_epochs=300
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
