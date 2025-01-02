import argparse

import torch
import torch.multiprocessing as mp

from decoder_finetune_trainer import DecoderFinetuneTrainer


def train_process(rank):
    print(f"Process {rank} started")
    print(f"CUDA Available? {torch.cuda.is_available()}")

def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Train a decoder finetuning model.")

    # Required arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to finetune (e.g., 'google/flan-t5-small').")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model checkpoints and logs.")

    # Optional arguments with default values
    parser.add_argument("--max_length", type=int, default=32,
                        help="Maximum sequence length for the model (default: 32).")
    parser.add_argument("--data_folder", type=str, default="eng-literal",
                        help="Folder containing the training and validation data (default: 'eng-literal').")
    parser.add_argument("--train_samples", type=int, default=100,
                        help="Number of training samples to use (default: 100).")
    parser.add_argument("--val_samples", type=int, default=10,
                        help="Number of validation samples to use (default: 10).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and validation (default: 8).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer (default: 1e-4).")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs to train (default: 50).")
    parser.add_argument("--wandb_run_name", type=str, default="decoder_finetuning",
                        help="Name of the wandb run (default: 'decoder_finetuning').")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a checkpoint to resume training (default: None).")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    mp.set_start_method("spawn", force=True)
    mp.spawn(train_process, args=(), nprocs=4)

    trainer = DecoderFinetuneTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        data_folder=args.data_folder,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        wandb_run_name=args.wandb_run_name,
        checkpoint_path=args.checkpoint_path
    )

    # Start training
    trainer.train()