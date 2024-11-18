# Metadata
__version__ = "0.0.1"
__author__ = "Yiyi Chen"

from .InversionModel import EmbeddingInverter
from .alignment_models import (LinearAligner,
                               optimal_transport_align,
                               procrustes_alignment
                               )

# Initialize package-level settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Package initialized")
