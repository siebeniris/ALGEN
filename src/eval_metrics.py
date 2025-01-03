from collections import defaultdict

import numpy as np
import torch
from torch.nn import MSELoss, CosineEmbeddingLoss
from rouge_score import rouge_scorer

from inversion_utils import decode_embeddings


rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
mse_loss = MSELoss()
cos_loss = CosineEmbeddingLoss()
cos = torch.nn.CosineSimilarity(dim=1)



def get_rouge_scores(pred, target):
    metrics_test = defaultdict(list)
    for pred, target in zip(pred, target):
        scores = rouge_scorer.score(target, pred)
        for metric, score in scores.items():
            metrics_test[f"{metric}_f"].append(score.fmeasure)
    return metrics_test


def loss_metrics(X, Y):
    X_reshape= X.view(-1, X.size(-1))
    Y_reshape= Y.view(-1, Y.size(-1))
    batch_seq_len = Y_reshape.shape[0]
    device = X.device
    target = torch.ones(batch_seq_len).to(device)

    cosloss = cos_loss(X_reshape, Y_reshape, target, reduction='mean')
    mseloss = mse_loss(X, Y, reduction='mean') # already averaged
    return cosloss, mseloss


def eval_embeddings(X, Y):
    """
    Evaluate the embeddings cosine similarities and mse loss.
    :param X:
    :param Y:
    :return:
    """
    return cos(X,Y).mean(), mse_loss(X,Y)



def eval_decoding(X_test_aligned, Y_test, Y_test_gold_texts, Y_test_mask, target_model, target_tokenizer,
                  num_beams,
                  do_sample,
                  repetition_penalty,
                  length_penalty,
                  top_k, top_p, temperature,
                  max_length):
    # under the condition of the same tokenization
    X_test_output = decode_embeddings(X_test_aligned, Y_test_mask,
                                      target_model, target_tokenizer,
                                      num_beams,
                                      do_sample,
                                      repetition_penalty,
                                      length_penalty,
                                      top_k, top_p, temperature,
                                      None, max_length=max_length)

    Y_test_output = decode_embeddings(Y_test, Y_test_mask,
                                      target_model, target_tokenizer,
                                      num_beams,
                                      do_sample,
                                      repetition_penalty,
                                      length_penalty,
                                      top_k, top_p, temperature,
                                      None, max_length=max_length)

    print(X_test_output[:10])
    print(Y_test_output[:10:])
    print(Y_test_gold_texts[:10])
    print("*" * 40)

    # TODO: compartmentalize the function and evaluate with more metrics
    # ADD also translation here.

    # get rouge scores.
    results_dict = defaultdict(dict)
    rouge_score_X_vs_gold = get_rouge_scores(X_test_output, Y_test_gold_texts)
    rouge_score_Y_vs_gold = get_rouge_scores(Y_test_output, Y_test_gold_texts)
    rouge_score_X_vs_Y = get_rouge_scores(X_test_output, Y_test_output)

    results_dict["X_vs_gold"] = {k: np.mean(v) for k, v in rouge_score_X_vs_gold.items()}
    results_dict["Y_vs_gold"] = {k: np.mean(v) for k, v in rouge_score_Y_vs_gold.items()}
    results_dict["X_vs_Y"] = {k: np.mean(v) for k, v in rouge_score_X_vs_Y.items()}

    return results_dict, X_test_output, Y_test_output