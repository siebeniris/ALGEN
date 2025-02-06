import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import evaluate
from rouge_score import rouge_scorer
from tqdm import tqdm
from deep_translator import GoogleTranslator

print("Initializing the metrics BLEU and ROUGE...")
bleu_metric = evaluate.load("sacrebleu")
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def get_rouge_scores(pred, target):
    metrics_test = defaultdict(list)
    for pred, target in zip(pred, target):
        scores = rouge_scorer.score(target, pred)
        for metric, score in scores.items():
            metrics_test[f"{metric}_f"].append(score.fmeasure)
    return metrics_test


rouge_metric = get_rouge_scores

source_model_dict = {'google-bert_bert-base-multilingual-cased': 'mBERT',
                     # "sentence-transformers/all-MiniLM-L6-v2": 'SBERT',
                     'text-embedding-ada-002': 'OpenAI (ada-2)',
                     # 'text-embedding-3-large':'OpenAI (3-large)',
                     'google-t5_t5-base': 'T5',
                     # 'google/flan-t5-base': 'FlanT5-Base',
                     'google_mt5-base': 'mT5',
                     # 'intfloat/multilingual-e5-base': 'ME5',
                     'sentence-transformers_gtr-t5-base': 'GTR'}


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
        "rougeL": round(rouge_score_dict["rougeL_f"], 4) * 100,
        "rouge1": round(rouge_score_dict["rouge1_f"], 4) * 100,
        "rouge2": round(rouge_score_dict["rouge2_f"], 4) * 100,
        "exact_match": round(sum(exact_matches) / len(exact_matches), 4)
    }
    # oracle : 70 rougeL
    return gen_metrics


# recording results
model_name_ls =[]
rougeL_translated = []
cos_original = []
rougel_orignal= []
attack_langs =[]
target_langs =[]

counter = 0
for attack_dataset_name in ["mmarco_english", "mmarco_german", "mmarco_french", "mmarco_spanish"]:
    attack_lang = attack_dataset_name.replace("mmarco_", "")
    print(f"attack language {attack_lang}")

    result_dir = f"outputs/google_flan-t5-small/yiyic_{attack_dataset_name}_maxlength32_train150000_batch_size128_lr0.0001_wd0.0001_epochs100"
    for subfolder in os.listdir(result_dir):
        filepath = os.path.join(result_dir, subfolder, "results_texts.csv")

        translation_outputpath = filepath.replace("_texts.csv", "_translated.csv")
        eval_result_outputpath = filepath.replace("_texts.csv", "_translated.json")

        if not os.path.exists(eval_result_outputpath):

            matching_keys = [key for key in source_model_dict if key in filepath]

            # Print result
            if matching_keys:
                print(f"Matching keys found in path: {matching_keys}")

                basedir = filepath.split("/")[-2]
                if basedir.startswith("attack_") and basedir.endswith("train1000"):
                    dataset_lang_parts = basedir.replace("attack_yiyic_", "").split("_")

                    # target language
                    if dataset_lang_parts[0] == "mmarco":
                        target_lang = dataset_lang_parts[1]

                        # only do crosslingual
                        if attack_lang != target_lang:
                            print(f"Initializing translator {attack_lang} => {target_lang}")
                            translator = GoogleTranslator(source="auto", target=target_lang)

                            with open(os.path.join(result_dir, subfolder, "results.json")) as f:
                                result_dict = json.load(f)

                                # recording results
                            model_name_ls.append(matching_keys[0])
                            rougel_orignal.append(result_dict["test_results"]["rougeL"])
                            cos_original.append(result_dict["loss"]["X_Y_test_COS"])
                            attack_langs.append(attack_lang)
                            target_langs.append(target_lang)

                            with open(filepath) as f:
                                results_df = pd.read_csv(f)

                            # translate the predicted texts.
                            prediction_texts = results_df["predictions"].tolist()
                            target_texts = results_df["reference"].tolist()

                            # translate
                            translated_batch = []
                            for text in tqdm(prediction_texts):
                                translated_text = translator.translate(text)
                                translated_batch.append(translated_text)

                            eval_result = eval_texts(translated_batch, target_texts)

                            results_df["translated_predictions"] = translated_batch
                            print(results_df.head(2))
                            results_df.to_csv(translation_outputpath)

                            print(f"writing to results to {eval_result_outputpath}")
                            print(eval_result)
                            with open(eval_result_outputpath, "w") as f:
                                json.dump(eval_result, f)

                            rougeL_translated.append(eval_result["rougeL"])

                            counter += 1

            else:
                print("No matching keys found in path.")
        else:
            print(f"{eval_result_outputpath} exists!")

print(counter)

df_result_all = pd.DataFrame({
    "Victim Model": model_name_ls,
    "RougeL(Original)": rougel_orignal,
    "RougeL(Trans)": rougeL_translated,
    "Attack Langs": attack_langs,
    "Victim Langs": target_langs
})

df_result_all.to_csv("outputs/crosslingual_eval_results.csv", index=False)
