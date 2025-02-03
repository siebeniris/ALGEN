from trainerConfig import TrainerConfig
from InversionTrainer import EmbeddingInverterTrainer


def get_ot_config_variants():
    """Different OT configurations for various scenarios"""

    configs = {
        "default": {
            # Balanced regularization for general use
            "ot_reg": 0.1,  # Entropic regularization
            "ot_reg_m": 10.0  # Mass regularization
        },
        "more_precise": {
            # Lower regularization for more precise alignment
            "ot_reg": 0.01,  # Less smoothing
            "ot_reg_m": 20.0  # Stronger mass conservation
        },
        "more_flexible": {
            # Higher regularization for more flexible alignment
            "ot_reg": 0.5,  # More smoothing
            "ot_reg_m": 5.0  # More flexible mass distribution
        },
        "computational_efficient": {
            # Higher regularization for faster computation
            "ot_reg": 1.0,  # Much more smoothing
            "ot_reg_m": 1.0  # Very flexible mass distribution
        }
    }
    return configs


def explain_ot_parameters():
    """Explanation of OT parameters and their effects"""
    explanations = {
        "ot_reg (Entropic Regularization)": {
            "purpose": "Controls the smoothness of the transport plan",
            "effects": {
                "low_values": [
                    "More precise/sparse transport plan",
                    "Better preservation of local structure",
                    "Slower convergence",
                    "May be less stable"
                ],
                "high_values": [
                    "Smoother/more diffused transport plan",
                    "Faster convergence",
                    "More stable optimization",
                    "May lose fine-grained details"
                ],
                "typical_range": "0.01 to 1.0",
                "recommended_starting_point": "0.1"
            }
        },
        "ot_reg_m (Mass Regularization)": {
            "purpose": "Controls how strictly mass conservation is enforced",
            "effects": {
                "low_values": [
                    "More flexible mass distribution",
                    "Better handling of different sequence lengths",
                    "May lose some structural correspondence"
                ],
                "high_values": [
                    "Stricter mass conservation",
                    "Better preservation of token-level correspondence",
                    "May be too rigid for very different sequences"
                ],
                "typical_range": "1.0 to 20.0",
                "recommended_starting_point": "10.0"
            }
        }
    }
    return explanations


# Example configuration selection based on use case
def select_ot_config(scenario: str):
    """Select OT configuration based on specific needs"""
    configs = {
        "default": TrainerConfig(
            align_method="ot",
            ot_reg=0.1,
            ot_reg_m=10.0
        ),
        "cross_lingual": TrainerConfig(
            align_method="ot",
            ot_reg=0.5,  # More flexible for different languages
            ot_reg_m=5.0  # Less strict mass conservation for different sentence structures
        ),
        "same_language": TrainerConfig(
            align_method="ot",
            ot_reg=0.01,  # More precise alignment
            ot_reg_m=20.0  # Stricter token correspondence
        ),
        "long_sequences": TrainerConfig(
            align_method="ot",
            ot_reg=0.3,  # Balance between precision and flexibility
            ot_reg_m=7.0  # Moderate mass conservation
        )
    }
    return configs.get(scenario, configs["default"])


def grid_search_ot_params():
    """Grid search parameters for OT hyperparameter tuning"""
    return {
        "ot_reg": [0.01, 0.1, 0.5, 1.0],
        "ot_reg_m": [1.0, 5.0, 10.0, 20.0]
    }


# Example usage of grid search
def run_ot_grid_search(base_config: TrainerConfig, train_texts, eval_texts):
    """Run grid search for OT parameters"""
    params = grid_search_ot_params()
    results = []

    for reg in params["ot_reg"]:
        for reg_m in params["ot_reg_m"]:
            config = base_config
            config.ot_reg = reg
            config.ot_reg_m = reg_m

            trainer = EmbeddingInverterTrainer(**config.to_dict())
            metrics = trainer.train(train_texts, eval_texts)

            results.append({
                "ot_reg": reg,
                "ot_reg_m": reg_m,
                "metrics": metrics
            })

    return results