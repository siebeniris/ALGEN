import os
import sys

import torch
import numpy as np
import ot
import json
from rouge_score import rouge_scorer
from tqdm import tqdm
import itertools
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, AutoModel, AutoTokenizer
import torch.nn as nn
from torch.nn import LayerNorm
import transformers.models.t5.modeling_t5 as t5_modeling

t5_modeling.T5LayerNorm = LayerNorm

device = "cuda" if torch.cuda.is_available() else "cpu"
# set up rouge scores
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
cos = torch.nn.CosineSimilarity(dim=1)


def load_data(
        data_path="/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus/eng-literal/train.txt"
):
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = [x.replace("\n", "") for x in f.readlines()]
    print(f"Data length {len(data)}")
    # data_sampled = data[:10000]+data[-300:]
    return data[:5000], data[-300:]


def decode_embeddings(embeddings,
                      attention_mask,
                      model_G, tokenizer_G,
                      num_beams,
                      do_sample,
                      repetition_penalty,
                      length_penalty,
                      top_k, top_p, temperature, np_repeat_ngram_size,
                      decoder_input_ids=None, max_length=32):
    """Decode embeddings back to text."""
    with torch.no_grad():
        embeddings = embeddings.to(torch.float32)
        batch_size, seq_length, hidden_size = embeddings.size()
        print(f"embeddings mean: {embeddings.mean().item()}, std: {embeddings.std().item()}")

        if attention_mask == None:
            print(f"all ones for attention_mask")
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=embeddings.device)

        if embeddings.size(0) == 0 or attention_mask.size(0) == 0:
            print("Error: Empty embeddings or attention mask.")
            return ["Error during generation: Empty input"] * batch_size

        # Create encoder outputs
        # wrap embeddings in BaseModelOutput to simulate encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)
        print(f"Encoder outputs last_hidden_state shape: {encoder_outputs.last_hidden_state.shape}")

        # Initialize decoder input IDs
        if decoder_input_ids == None:
            print("creating decoder_input_ids:")
            decoder_input_ids = torch.full(
                (batch_size, 1),
                1,  # tokenizer_G.eos_token_id, # start decoding with the EOS,
                dtype=torch.long,
                device=device
            )
        print("decoder input_ids:", decoder_input_ids.shape)

        assert embeddings.size(0) == decoder_input_ids.size(0), "Decoder input batch size mismatch!"

        # print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
        try:

            # Generate output
            generated = model_G.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # initialized with [1], eos for the whole batch.
                max_length=max_length + 10,
                # no_repeat_ngram_size
                num_beams=num_beams,  # Increased beam size
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                no_repeat_ngram_size=np_repeat_ngram_size,
                early_stopping=True,
                pad_token_id=tokenizer_G.pad_token_id,
                eos_token_id=tokenizer_G.eos_token_id,
            )

            # decoded_text = tokenizer_G.batch_decode(
            #     generated, skip_special_tokens=False, clean_up_tokenization_spaces=True
            # )

            decoded_text = tokenizer_G.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            decoded_text = [text.strip() for text in decoded_text]

            # print(f"decoded text:", decoded_text)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            decoded_text = ["Error during generation"] * batch_size
    return decoded_text


def add_punctuation_to_decoded(decoded_texts, punctuation="."):
    processed_texts = []
    for text in decoded_texts:
        if not text.endswith(punctuation):
            text += " " + punctuation
        processed_texts.append(text)
    return processed_texts


def preprocess_sentence(sentence, max_length, tokenizer, punctuation="."):
    sentence = add_punctuation_to_decoded(sentence, punctuation)
    tokenized = tokenizer(sentence, padding="max_length", truncation=True,
                          max_length=max_length, return_tensors="pt")
    return tokenized


def adding_punctuation_to_tokenization(samples, tokenizer, max_length):
    # adding punctuation in tokens.
    tokenized_batch = preprocess_sentence(samples, max_length=max_length - 2, tokenizer=tokenizer)
    decoded_batch = [tokenizer.decode(tb, skip_special_tokens=True) for tb in tokenized_batch["input_ids"]]
    retokenized_batch = preprocess_sentence(decoded_batch, max_length=max_length, tokenizer=tokenizer)
    return retokenized_batch


def fill_in_pad_eos_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return tokenizer


def load_tokenizer_models(source_model_name, target_model_name):
    # the goal is to align source model to target model.
    # the target model should be a encoder-decoder model
    # the source model only need embeddings
    source_model = AutoModel.from_pretrained(source_model_name)
    target_model = T5ForConditionalGeneration.from_pretrained(target_model_name)

    source_tokenizer = AutoTokenizer.from_pretrained(source_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    fill_in_pad_eos_token(source_tokenizer)
    fill_in_pad_eos_token(target_tokenizer)

    source_model.resize_token_embeddings(len(source_tokenizer))
    target_model.resize_token_embeddings(len(target_tokenizer))

    return source_model, target_model, source_tokenizer, target_tokenizer


def get_embeddings(train_data, test_data,
                   source_model, target_model,
                   source_tokenizer, target_tokenizer, max_length=32):
    X_tokens = adding_punctuation_to_tokenization(train_data, source_tokenizer, max_length)
    Y_tokens = adding_punctuation_to_tokenization(train_data, target_tokenizer, max_length)
    X_test_tokens = adding_punctuation_to_tokenization(test_data, source_tokenizer, max_length)
    Y_test_tokens = adding_punctuation_to_tokenization(test_data, target_tokenizer, max_length)

    # X_tokens = source_tokenizer(train_data, padding="max_length", truncation=True,
    #                             max_length=max_length, return_tensors="pt")
    # Y_tokens = target_tokenizer(train_data, padding="max_length", truncation=True,
    #                             max_length=max_length, return_tensors="pt")
    # X_test_tokens = source_tokenizer(test_data, padding="max_length", truncation=True,
    #                                  max_length=max_length, return_tensors="pt")
    # Y_test_tokens = target_tokenizer(test_data, padding="max_length", truncation=True,
    #                                  max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        X = source_model.encoder(**X_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y = target_model.encoder(**Y_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        X_test = source_model.encoder(**X_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        Y_test = target_model.encoder(**Y_test_tokens).last_hidden_state  # Shape: (batch, seq_len, hidden_size)

    return X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens


def mapping_X_to_Y(X, Y):
    batch_size, seq_len, _ = X.shape
    X_ = X.reshape(-1, X.shape[-1])
    Y_ = Y.reshape(-1, Y.shape[-1])
    As = torch.linalg.pinv(X_.T @ X_) @ X_.T @ Y_
    Xs = X_ @ As

    Xs_ = Xs.view(batch_size, seq_len, Xs.shape[-1])
    return cos(Xs, Y_).mean(), Xs_, As


def test_alignment(X_test, Y_test, As):
    test_batch_size, test_seq_len, test_x_dim = X_test.shape

    x = X_test.reshape(-1, X_test.shape[-1])
    y = Y_test.reshape(-1, Y_test.shape[-1])
    x_ = x @ As
    return cos(x_, y).mean(), x_.view(test_batch_size, test_seq_len, x_.shape[-1])


def get_rouge_scores(pred, target):
    metrics_test = defaultdict(list)
    for pred, target in zip(pred, target):
        scores = rouge_scorer.score(target, pred)
        for metric, score in scores.items():
            metrics_test[f"{metric}_f"].append(score.fmeasure)
    # for k,v in metrics_test.items():
    #     print(k, np.mean(v))
    return metrics_test


def eval_decoding(X_test_aligned, Y_test, Y_test_gold_texts, Y_test_mask, target_model, target_tokenizer,
                  num_beams,
                  do_sample,
                  repetition_penalty,
                  length_penalty,
                  top_k, top_p, temperature, np_repeat_ngram_size,
                  max_length):
    # under the condition of the same tokenization
    X_test_output = decode_embeddings(X_test_aligned, Y_test_mask,
                                      target_model, target_tokenizer,
                                      num_beams,
                                      do_sample,
                                      repetition_penalty,
                                      length_penalty,
                                      top_k, top_p, temperature, np_repeat_ngram_size,
                                      None, max_length=max_length)

    Y_test_output = decode_embeddings(Y_test, Y_test_mask,
                                      target_model, target_tokenizer,
                                      num_beams,
                                      do_sample,
                                      repetition_penalty,
                                      length_penalty,
                                      top_k, top_p, temperature, np_repeat_ngram_size,
                                      None, max_length=max_length)

    print(X_test_output[:10])
    print(Y_test_output[:10:])
    print(Y_test_gold_texts[:10])
    print("*" * 40)

    results_dict = defaultdict(dict)
    rouge_score_X_vs_gold = get_rouge_scores(X_test_output, Y_test_gold_texts)
    rouge_score_Y_vs_gold = get_rouge_scores(Y_test_output, Y_test_gold_texts)
    rouge_score_X_vs_Y = get_rouge_scores(X_test_output, Y_test_output)

    results_dict["X_vs_gold"] = {k: np.mean(v) for k, v in rouge_score_X_vs_gold.items()}
    results_dict["Y_vs_gold"] = {k: np.mean(v) for k, v in rouge_score_Y_vs_gold.items()}
    results_dict["X_vs_Y"] = {k: np.mean(v) for k, v in rouge_score_X_vs_Y.items()}

    return results_dict


def aligning_and_testing(source_model_name, target_model_name, max_length=32,
                         outputdir="results"):
    # SET UP rouge scorer and cosine similarity
    train_data, test_data = load_data()
    train_data = train_data[:1000]
    print(f"loading the models and tokenizers for source {source_model_name} and target {target_model_name}")
    source_model, target_model, source_tokenizer, target_tokenizer \
        = load_tokenizer_models(source_model_name, target_model_name)

    outputdir = os.path.join(outputdir,
                             f"{source_model_name.replace("/", "-")}_to_{target_model_name.replace("/", "-")}")
    os.makedirs(outputdir, exist_ok=True)

    X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens = get_embeddings(
        train_data, test_data,
        source_model, target_model,
        source_tokenizer, target_tokenizer, max_length=max_length)

    print(f"X {X.shape}, Y {Y.shape}, X_test {X_test.shape}, Y_test {Y_test.shape}")
    print("Mappting X to Y...")
    X_Y_cossim, Xs, T = mapping_X_to_Y(X, Y)
    print(f"Cosine similarity between aligned X and Y {X_Y_cossim} ...")

    # directly from look-up dictionary from tokenizers, as the gold standard for both X_test and Y_test
    Y_test_gold = [target_tokenizer.decode(tb, skip_special_tokens=True)
                   for tb in Y_test_tokens["input_ids"]]

    X_Y_TEST_COSSIM, X_test_aligned = test_alignment(X_test, Y_test, T)

    # print("Saving vectors and ids ...")
    # tensor_dict = {
    #     "X": X, "Y": Y, "X_test": X_test, "Y_test": Y_test,
    #     "X_aligned": Xs, "T": T, "X_test_tokens": X_test_tokens, "Y_test_tokens": Y_test_tokens,
    #     "X_test_aligned": X_test_aligned
    # }
    #
    # torch.save(tensor_dict, os.path.join(outputdir, "X_Y_data.pth"))

    cosine_similarity_metrics = {
        "X_Y_COS": X_Y_cossim.detach().cpu().numpy().tolist(),
        "X_Y_test_COS": X_Y_TEST_COSSIM.detach().cpu().numpy().tolist()
    }
    print(cosine_similarity_metrics)
    with open(os.path.join(outputdir, "cosine_similarities.json"), "w+") as f:
        json.dump(cosine_similarity_metrics, f)

    grid_params = {
        "num_beams": [3],  # Beam search, deterministic
        "repetition_penalty": [1.7, 1.9, 2.0],  # TODO: change this to higher.
        "length_penalty": [0.8, 1.6, 1.8, 2.0],  # TODO: change this to higher. Adjust sequence length preference
        "do_sample": [False],
        "top_k": [None],
        "top_p": [None],
        "temperature": [None],
        "np_repeat_ngram_size": [2, 3, 4]
        # "top_k": [None, 20, 50, 80, 100],
        # "top_p": [None, 0.8, 0.9],
        # "temperature": [None, 0.7, 1.0, 1.2, 1.5]
    }

    print(f"grid search on decoding strategies....")

    param_combinations = list(itertools.product(
        grid_params["num_beams"],
        grid_params["repetition_penalty"],
        grid_params["length_penalty"],
        grid_params["do_sample"],
        grid_params["top_k"],
        grid_params["top_p"],
        grid_params["temperature"],
        grid_params["np_repeat_ngram_size"]
    ))

    valid_combinations = []
    for combo in param_combinations:
        # Unpack parameters
        num_beams, repetition_penalty, length_penalty, do_sample, top_k, top_p, temperature, np_repeat_ngram_size = combo

        # Apply constraints
        if not do_sample and (top_k or top_p or temperature):  # Ignore sampling parameters if do_sample=False
            continue
        if num_beams > 1 and do_sample and top_k is not None:  # Sampling works better with fewer beams
            continue

        # Add valid combination
        valid_combinations.append({
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "do_sample": do_sample,
            "top_k": top_k if do_sample else None,
            "top_p": top_p if do_sample else None,
            "temperature": temperature if do_sample else None,
            "np_repeat_ngram_size": np_repeat_ngram_size
        })
    print(f"valid combinations: {len(valid_combinations)}")

    ###### TODO:
    ### SINCE ROUGE-2 IS NOT IMPROVING, WE NEED TO ADD
    # no_repeat_ngram_size, length_penalty, max_new_tokens =50
    # increase length_penalty >=1.2 could be experimented with
    # no_repeat_ngram_size >=2
    for params in valid_combinations:
        num_beams, repetition_penalty, length_penalty, do_sample, top_k, top_p, temperature, np_repeat_ngram_size = \
            (params["num_beams"], params["repetition_penalty"], params["length_penalty"],
             params["do_sample"], params["top_k"], params["top_p"], params["temperature"],
             params["np_repeat_ngram_size"])

        print(f" Decoding with the strategy num_beams {num_beams}, do_sample {do_sample},"
              f" repetition {repetition_penalty}, length penalty {length_penalty},  "
              f"top k {top_k}, top p {top_p}, temperature {temperature}, no repeat ngram size {np_repeat_ngram_size}")
        print(f"Cosine Similarity between aligned X_test and Y_test {X_Y_TEST_COSSIM}")

        if not do_sample:
            outputfile = os.path.join(outputdir,
                                      f"rouge_score-num_beams_{num_beams}-do_sample_{do_sample}"
                                      f"-repetition_{repetition_penalty}-length_{length_penalty}"
                                      f"-no_repeat_ngrams_{np_repeat_ngram_size}.json")
        else:
            outputfile = os.path.join(outputdir,
                                      f"rouge_score-num_beams_{num_beams}-do_sample_{do_sample}-repetition_{repetition_penalty}"
                                      f"-length_{length_penalty}-topk_{top_k}-topp_{top_p}-temp_{temperature}.json")

        if not os.path.exists(outputfile):
            rouge_result_dict = eval_decoding(X_test_aligned, Y_test, Y_test_gold,
                                              Y_test_tokens["attention_mask"],
                                              target_model,
                                              target_tokenizer,
                                              num_beams,
                                              do_sample,
                                              repetition_penalty,
                                              length_penalty,
                                              top_k, top_p, temperature,
                                              np_repeat_ngram_size,
                                              max_length)

            print(f"recording the results to {outputdir}")

            with open(outputfile, "w") as f:
                json.dump(rouge_result_dict, f)
        else:
            print(f"{outputfile} exists..")


if __name__ == '__main__':
    import plac

    plac.call(aligning_and_testing)
