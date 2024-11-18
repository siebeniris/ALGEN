from typing import Callable, Dict

import torch
import transformers

from src.models import FewShotInversionModel

def tokenize_function(
        tokenizer: transformers.PreTrainedTokenizer,
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        text_column_name: str,
        max_seq_length: int,
        padding: bool = False,
        prefix: str = None,
        lang_id: bool = False,
        script_id: bool = False,
) -> Callable[[Dict], Dict]:
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if prefix:
            if lang_id and not script_id:
                examples[text_column_name] = [f"{prefix}: [{lang.split('_')[0]}] {text}" for text, lang in
                                              zip(examples[text_column_name],
                                                  examples["lang"])]
            elif lang_id and script_id:
                examples[text_column_name] = [f"{prefix}: [{lang.split('_')[0]}] [{lang.split('_')[1]}] {text}" for
                                              text, lang in
                                              zip(examples[text_column_name],
                                                  examples["lang"])]
            else:
                examples[text_column_name] = [f"{prefix}: {text}" for text in examples[text_column_name]]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        encoder_output = {f"encoder_{k}": v for k, v in output.items()}

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}



        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output, **encoder_output}

    return tokenize_function_inner


def embed_dataset_batch(model: FewShotInversionModel, batch: Dict) -> Dict:
    # TODO: check this out
    assert "embedder_input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert "encoder_input_ids" in batch.keys(), f"no encoder input ids"
    # changed
    assert hasattr(model, "align_embeddings")

    encoder_input_ids = batch["encoder_input_ids"].to(next(model.parameters()).device)
    encoder_attention_mask = batch["encoder_attention_mask"].to(next(model.parameters()).device)
    embedder_input_ids = batch["embedder_input_ids"].to(next(model.parameters()).device)
    embedder_attention_mask = batch["embedder_attention_mask"].to(next(model.parameters()).device)

    # Create the embeddings without gradients
    with torch.no_grad():
        # Pass the embedder input IDs and attention mask to align_embeddings
        frozen_embeddings = model.align_embeddings(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_masks=embedder_attention_mask,
            encoder_input_ids=encoder_input_ids,
            encoder_attention_masks=encoder_attention_mask
        )

    batch["frozen_embeddings"] = frozen_embeddings
    return batch

