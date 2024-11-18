import logging
import os
import random
from typing import Dict, List

import yaml
import datasets
import torch

from src.run_args import DataArguments

def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    pass

def load_mt_ms_test() -> datasets.DatasetDict:
    """
    Multilingual multi-script test dataset.
    """
    test_dataset = datasets.load_dataset("yiyic/mt_ms_test")
    return test_dataset


def load_standard_val_datasets() -> datasets.DatasetDict:
    d = load_mt_ms_test()
    return d