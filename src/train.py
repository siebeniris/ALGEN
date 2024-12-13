import argparse
from trainerConfig import (
    TrainerConfig,
    get_default_config,
    get_ot_config,
    get_linear_config
)
from InversionTrainer import EmbeddingInverterTrainer
import yaml
import os
import torch

def save_config(config: TrainerConfig, save_dir: str):
    """Save configuration to YAML file"""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, f'config_{config.align_method}_epoch{config.num_epochs}.yaml')

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Saved configuration to {config_path}")


def load_config_from_yaml(config_path: str) -> TrainerConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return TrainerConfig(**config_dict)


def train_with_config(config: TrainerConfig):
    """Train model using provided configuration"""
    # Initialize trainer with config
    trainer = EmbeddingInverterTrainer(
        model_G_name=config.model_G_name,
        model_S_name=config.model_S_name,
        save_dir=config.save_dir,
        # resume training
        checkpoint_path=config.checkpoint_path,
        resume_training=config.resume_training,
        use_wandb=config.use_wandb,
        align_method=config.align_method,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        max_length=config.max_length,
        decoding_strategy=config.decoding_strategy,
        dataset_name=config.dataset_name,
        language_script=config.language_script,
        train_samples=config.train_samples,
    )
    # Save config before training
    save_config(config, config.save_dir)

    # Train model
    trainer.train()


def train_process(rank):
    print(f"Process {rank} started")
    print(f"CUDA Available? {torch.cuda.is_available()}")


def main():
    # This ensures the spawn method is used when initializing workers in the DataLoader.
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    mp.spawn(train_process, args=(), nprocs=4)

    parser = argparse.ArgumentParser(description='Train Embedding Inverter')
    parser.add_argument('--config', type=str, choices=[
        'default', 'ot'
    ], default='default', help='Configuration preset to use')
    parser.add_argument('--config_path', type=str, help='Path to custom YAML config file')

    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Give a checkpoint path to resume training")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training or not True or False")

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--align_method', type=str, default='ot', help='Alignment method to use')
    parser.add_argument('--train_samples', type=int, default=1000, help='Number of training samples to use')

    parser.add_argument("--target_model", type=str, default="t5-small")  # G-model
    parser.add_argument("--source_model", type=str, default="intfloat/multilingual-e5-small")  # S-model

    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()

    # Load configuration
    if args.config_path:
        config = load_config_from_yaml(args.config_path)
        print("Loaded custom configuration from", args.config_path)
    else:
        # Get preset configuration
        configs = {
            'default': get_default_config,
            'ot': get_ot_config,
            'linear': get_linear_config,
        }
        config = configs[args.config]()
        print(f"Using {args.config} configuration")

    # override config with CLI arguments
    config.save_dir = args.save_dir
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.align_method = args.align_method
    config.train_samples = args.train_samples
    config.use_wandb = args.use_wandb
    config.checkpoint_path = args.checkpoint_path
    config.resume_training = args.resume_training
    config.model_G_name = args.target_model
    config.model_S_name = args.source_model


    # Print configuration
    print("\nTraining Configuration:")
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")

    # Start training
    train_with_config(config)


if __name__ == "__main__":
    main()