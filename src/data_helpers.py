import logging
import os
import random
from typing import Dict, List

import yaml
import datasets
import torch





def load_mt_ms_test() -> datasets.DatasetDict:
    """
    Multilingual multi-script test dataset.
    """
    test_dataset = datasets.load_dataset("yiyic/mt_ms_test")
    return test_dataset


