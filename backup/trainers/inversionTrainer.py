from typing import Dict
import torch
import torch.nn as nn
import transformers
import math

from src.metrics_utils import CosineSimilarityLoss
from src.trainers.base import BaseTrainer


class FewShotInversionTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.encoder_decoder = self.model.encoder_decoder
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.embedder = self.model.embedder
        self.align_embeddings = self.model.align_embeddings

    def generate(self, inputs: Dict[str, torch.Tensor], generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    # put to Base file.
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        emb_s = inputs['emb_s']
        emb_g = inputs['emb_g']

        aligned_emb_s = model(emb_s)
        loss_fn = CosineSimilarityLoss()
        loss = loss_fn(aligned_emb_s, emb_g)

        return (loss, aligned_emb_s) if return_outputs else loss

    def training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)

    def evaluation_loop(
            self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        return output


    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we added extra dropout to the model
        # for older version of the models.
        if {
            "embedding_transform.2.weight",
            "embedding_transform.2.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform.3.weight"] = state_dict.pop(
                "embedding_transform.2.weight"
            )
            state_dict["embedding_transform.3.bias"] = state_dict.pop(
                "embedding_transform.2.bias"
            )
        return state_dict
