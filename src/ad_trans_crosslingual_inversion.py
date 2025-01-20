import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm

from easynmt import EasyNMT
model = EasyNMT('opus-mt')

source_models = ['FlanT5-Base', 'T5', 'GTR', 'mT5', 'ME5', 'mBERT', 'SBERT', 'TEA2', 'TE3L']

source_model_dict = {'google-bert/bert-base-multilingual-cased': 'mBERT',
                     "sentence-transformers/all-MiniLM-L6-v2": 'SBERT',
                     'text-embedding-3-large': 'TE3L',
                     'text-embedding-ada-002': 'TEA2',
                     'google-t5/t5-base': 'T5',
                     'google/flan-t5-base': 'FlanT5-Base',
                     'google/mt5-base': 'mT5',
                     'intfloat/multilingual-e5-base': 'ME5',
                     'sentence-transformers/gtr-t5-base': 'GTR'}

dataset2lang = {"mmarco_english":"en", "mmarco_german":"de", "mmarco_french":"fr", "mmarco_spanish":"es"}

import evaluate
from rouge_score import rouge_scorer

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def get_rouge_scores(pred, target):
    metrics_test = defaultdict(list)
    for pred, target in zip(pred, target):
        scores = rouge_scorer.score(target, pred)
        for metric, score in scores.items():
            metrics_test[f"{metric}_f"].append(score.fmeasure)
    return metrics_test


bleu_metric = evaluate.load("sacrebleu")
rouge_metric = get_rouge_scores


def eval_texts(predictions, references):
    bleu_scores = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric(predictions, references)
    # print(bleu_scores)
    rouge_score_dict = {k: np.mean(v) for k, v in rouge_score.items()}
    exact_matches = np.array(predictions) == np.array(references)
    gen_metrics = {
        "bleu": round(bleu_scores["score"], 2),
        "bleu1": round(bleu_scores["precisions"][0], 2),
        "bleu2": round(bleu_scores["precisions"][1], 2),
        "bleu3": round(bleu_scores["precisions"][2], 2),
        "bleu4": round(bleu_scores["precisions"][3], 2),
        "rougeL": round(rouge_score_dict["rougeL_f"], 4),
        "rouge1": round(rouge_score_dict["rouge1_f"], 4),
        "rouge2": round(rouge_score_dict["rouge2_f"], 4),
        "exact_match": round(sum(exact_matches) / len(exact_matches), 4)
    }
    # oracle : 70 rougeL
    return gen_metrics


name2model = {v:k for k,v in source_model_dict.items()}

def translate_preds(preds, source_lang, target_lang):
    l = []
    for sentence in tqdm(preds):
        try:
            pred_sent = model.translate(sentence, source_lang=source_lang, target_lang=target_lang)
        except Exception:
            pred_sent = sentence
        l.append(pred_sent)
    return l


result_dir = "outputs/google_flan-t5-small/yiyic_mmarco_english_maxlength32_train150000_batch_size128_lr0.0001_wd0.0001_epochs100"

for attack_dataset_name in ["mmarco_english", "mmarco_german", "mmarco_french", "mmarco_spanish"]:
    attack_decoder_dir = f"outputs/google_flan-t5-small/yiyic_{attack_dataset_name}_maxlength32_train150000_batch_size128_lr0.0001_wd0.0001_epochs100/"
    for m, model_name in name2model.items():
        model_name_ = model_name.replace("/", "_")
        for victim_dataset_name in ["mmarco_english", "mmarco_german", "mmarco_french", "mmarco_spanish"]:
            target_lang = dataset2lang[victim_dataset_name]
            source_lang = dataset2lang[attack_dataset_name]

            if attack_dataset_name != victim_dataset_name:
                print(f"attack lang {attack_dataset_name}, victim lang {victim_dataset_name}")

                subdir = os.path.join(attack_decoder_dir, f"attack_yiyic_{victim_dataset_name}_{model_name_}_train1000")

                outputpath = os.path.join(subdir, "results_texts_trans.csv")
                if not os.path.exists(outputpath):
                    df_texts = pd.read_csv(os.path.join(subdir, "results_texts.csv"))
                    predictions = df_texts["predictions"].tolist()
                    references = df_texts["reference"].tolist()
                    print(f"translating to {target_lang}")
                    trans_preds = translate_preds(predictions, source_lang, target_lang)

                    trans_preds = [str(pred) if not isinstance(pred, str) else pred for pred in trans_preds]
                    references = [str(ref) if not isinstance(ref, str) else ref for ref in references]

                    print("eval...")
                    string_results = eval_texts(trans_preds, references)

                    df_texts_trans = pd.DataFrame({
                        "predictions": predictions,
                        "predictions_trans": trans_preds,
                        "reference": references
                    })
                    df_texts_trans.to_csv(os.path.join(subdir, "results_texts_trans.csv"), index=False)
                    print(f"writing results to {subdir}")
                    with open(os.path.join(subdir, "results_trans.json"), "w") as f:
                        json.dump(string_results, f)
                else:
                    print(f"{outputpath} exists")
