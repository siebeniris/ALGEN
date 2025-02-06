import os


source_model_dict = {'google-bert_bert-base-multilingual-cased': 'mBERT',
                     # "sentence-transformers/all-MiniLM-L6-v2": 'SBERT',
                     'text-embedding-ada-002': 'OpenAI (ada-2)',
                     # 'text-embedding-3-large':'OpenAI (3-large)',
                     'google-t5_t5-base': 'T5',
                     # 'google/flan-t5-base': 'FlanT5-Base',
                     'google_mt5-base': 'mT5',
                     # 'intfloat/multilingual-e5-base': 'ME5',
                     'sentence-transformers_gtr-t5-base': 'GTR'}




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

                            counter += 1

            else:
                print("No matching keys found in path.")
        else:
            print(f"{eval_result_outputpath} exists!")

print(counter)
