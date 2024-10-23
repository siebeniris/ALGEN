import collections
import copy
import logging
import os
import random
from time import time
# import statistics
from typing import Callable, Dict, List, Tuple, Union
import pandas as pd
import evaluate
import nltk
import numpy as np
import scipy.stats
import torch
import tqdm
import transformers
from src.metrics_utils import CosineSimilarityLoss


class InversionTrainer(transformers.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        emb_s = inputs['emb_s']
        emb_g = inputs['emb_g']

        aligned_emb_s = model(emb_s)
        loss_fn = CosineSimilarityLoss()
        loss = loss_fn(aligned_emb_s, emb_g)

        return (loss, aligned_emb_s) if return_outputs else loss

