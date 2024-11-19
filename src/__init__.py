# Metadata
__version__ = "0.0.1"
__author__ = "Yiyi Chen"

from .InversionModel import EmbeddingInverter
from .alignment_models import (LinearAligner,
                               optimal_transport_align,
                               procrustes_alignment
                               )
from .InversionTrainer import EmbeddingInverterTrainer
from .embeddingAlingerOT import EmbeddingAlignerOT
from .trainerConfig import (TrainerConfig,
                            get_default_config,
                            get_ot_config,
                            get_linear_config,
                            get_neural_config,
                            get_orthogonal_config)
from .create_dataset import EmbeddingDataset, custom_collate_fn


# Initialize package-level settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Package initialized")
