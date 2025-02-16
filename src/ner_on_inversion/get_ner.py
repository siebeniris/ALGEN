import os
import pandas as pd
import json

from tqdm import tqdm
from src.ner_on_inversion.eval_metrics import evaluate_ner

import stanza
import spacy_stanza

source_models = ['T5', 'GTR', 'mT5', 'mBERT', 'OpenAI(ada-2)']

source_model_dict = {'google-bert_bert-base-multilingual-cased': 'mBERT',
                     'text-embedding-ada-002': 'OpenAI(ada-2)',
                     'google-t5_t5-base': 'T5',
                     'google_mt5-base': 'mT5',
                     'sentence-transformers_gtr-t5-base': 'GTR'}

nlp = spacy_stanza.load_pipeline("en")

def get_ent_and_types(docs):
    docs_ent_l = []
    for doc in tqdm(nlp.pipe(docs)):
        ents_and_types = []
        for token in doc:
            if token.ent_type_:
                ents_and_types.append((token.text, token.ent_type_, token.ent_iob_))
                # print(token.text, token.ent_type_, token.ent_iob_)
        docs_ent_l.append(ents_and_types)
    return docs_ent_l


result_dir = "outputs/google_flan-t5-small/yiyic_multiHPLT_english_maxlength32_train150000_batch_size128_lr0.0001_wd0.0001_epochs100"
output_dir = "outputs/NER_multiHPLT_english"


for source_model, model_name in source_model_dict.items():
    resultpath = f"{result_dir}/attack_yiyic_multiHPLT_english_{source_model}_train1000/results_texts.csv"
    if os.path.exists(resultpath):
        df = pd.read_csv(resultpath)
        preds = df["predictions"].tolist()
        refers = df["reference"].tolist()
        preds_tokens = get_ent_and_types(preds)
        refers_tokens = get_ent_and_types(refers)



        print("eval ", source_model)
        eval_results = evaluate_ner(refers_tokens, preds_tokens)
        print(eval_results)
        output_path = os.path.join(output_dir, f"{model_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(eval_results, f)
        print("*"*20)



