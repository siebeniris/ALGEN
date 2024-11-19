from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    # Model configuration
    model_G_name: str = "t5-base"
    model_S_name: str = "t5-small"
    max_length: int = 128

    # Training configuration
    align_method: str = "linear"
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100

    # Checkpoint configuration
    save_dir: str = "checkpoints"
    checkpoint_path: Optional[str] = None
    resume_training: bool = False

    # Logging configuration
    use_wandb: bool = True

    # Alignment configuration
    adjust_weights_with_magnitude: bool = True
    ot_reg: float = 0.1
    ot_reg_m: float = 10.0

    # Generation configuration
    decoding_strategy: str = "beam"

    # Dataset configuration
    dataset_name: str = "flores"
    language_script: str = "eng_Latn"
    train_samples: int = 1000
    eval_samples: int = 200

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}


# Default configurations for different scenarios
def get_default_config():
    return TrainerConfig()


def get_linear_config():
    config = TrainerConfig()
    config.align_method = "linear"
    return config


def get_neural_config():
    config = TrainerConfig()
    config.align_method = "neural"
    return config


def get_orthogonal_config():
    config = TrainerConfig()
    config.align_method = "orthogonal"
    return config


def get_ot_config():
    config = TrainerConfig()
    config.align_method = "ot"
    config.adjust_weights_with_magnitude = True
    config.ot_reg = 0.1
    config.ot_reg_m = 10.0
    return config


if __name__ == '__main__':
    config = get_default_config()
    print("is instance")
    assert isinstance(config, TrainerConfig)
    assert isinstance(config.to_dict(), dict)

    config = get_linear_config()
    assert isinstance(config, TrainerConfig)