import os

import pandas as pd
import torch
import json
from torch.nn import LayerNorm
import transformers.models.t5.modeling_t5 as t5_modeling

from inversion_utils import (
    load_tokenizer_models,
    get_embeddings,
)
from inversion_methods import mapping_X_to_Y, test_alignment
from eval_metrics import eval_decoding
from utils import get_lang_file_dict
t5_modeling.T5LayerNorm = LayerNorm

device = "cuda" if torch.cuda.is_available() else "cpu"

# loading file dict for parallel bible data.
lang2files = get_lang_file_dict()


def aligning_and_testing(source_model, target_model,
                         source_tokenizer, target_tokenizer,
                         train_data, test_data,
                         outputdir,
                         max_length=32,
                         ):
    # SET UP rouge scorer and cosine similarity

    # if not os.path.exists(outputfile):
    X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens \
            = get_embeddings(
                train_data, test_data,
                source_model, target_model,
                source_tokenizer, target_tokenizer,
                max_length=max_length
                )

    print(f"X {X.shape}, Y {Y.shape}, X_test {X_test.shape}, Y_test {Y_test.shape}")
    print("Mapping X to Y.")
    X_Y_cossim, Xs, T = mapping_X_to_Y(X, Y)
    print(f"Cosine similarity between aligned X and Y {X_Y_cossim}.")

    # directly from look-up dictionary from tokenizers, as the gold standard for both X_test and Y_test
    Y_test_gold = [target_tokenizer.decode(tb, skip_special_tokens=True) for tb in Y_test_tokens["input_ids"]]

    X_Y_TEST_COSSIM, X_test_aligned = test_alignment(X_test, Y_test, T)

    cosine_similarity_metrics = {
        "X_Y_COS": X_Y_cossim.detach().cpu().numpy().tolist(),
        "X_Y_test_COS": X_Y_TEST_COSSIM.detach().cpu().numpy().tolist()
    }

    rouge_result_dict, X_test_output, Y_test_output \
        = eval_decoding(X_test_aligned, Y_test, Y_test_gold,
                                      Y_test_tokens["attention_mask"],
                                      target_model,
                                      target_tokenizer,
                                      num_beams=3,
                                      do_sample=False,
                                      repetition_penalty=2.0,
                                      length_penalty=2.0,
                                      top_k=None, top_p=None, temperature=None,
                                      max_length=max_length)

    print(f"recording the results to {outputdir}")
    result_dict = {
        "cosine_similarities": cosine_similarity_metrics,
        "rouge_results": rouge_result_dict
    }
    outputfile = os.path.join(outputdir, "eval_results.json")
    with open(outputfile, "w+") as f:
        json.dump(result_dict, f)

    df_output = pd.DataFrame({
        "X_output": X_test_output,
        "Y_output": Y_test_output,
        "Y_gold": Y_test_gold
    })

    df_output.to_csv(os.path.join(outputdir, "test_output.csv"), index=False)


def aligning_per_lang_to_flant5(output_dir="results"):
    lang_data_dir = "dataset/Morphology-Matters-corpus"
    source_model_names = ["google/flan-t5-base", "google-t5/t5-base",
                          "google/mt5-base",
                          "facebook/mbart-large-50"]
    target_model_names = ["google/flan-t5-small", "google/mt5-small"]
    # intialize model names.
    for source_model_name in source_model_names:
        for target_model_name in target_model_names:
            source_model_rename = source_model_name.replace("/", "-")
            target_model_rename = target_model_name.replace("/", "-")
            outputfolder = f"{source_model_rename}_to_{target_model_rename}"
            outputdir = os.path.join(output_dir, outputfolder)
            os.makedirs(outputdir, exist_ok=True)

            print(f"aligning {source_model_name} embedding to {target_model_name}")

            # iterate languages
            for lang, folderpath in lang2files.items():
                print(f"aligning language {lang}")

                lang_data_dir_ = os.path.join(lang_data_dir, folderpath)
                with open(os.path.join(lang_data_dir_, "train.txt")) as f:
                    train_data = [x.replace("\n", "") for x in f.readlines()]

                with open(os.path.join(lang_data_dir_, "test.txt")) as f:
                    test_data = [x.replace("\n", "") for x in f.readlines()][:200]

                source_model, target_model, source_tokenizer, target_tokenizer \
                    = load_tokenizer_models(source_model_name, target_model_name)

                for train_samples in [100, 500, 1000]:
                    print(f"There are {train_samples} samples and {len(test_data)} samples")
                    outputdir_ = os.path.join(outputdir, f"{lang}_samples_{train_samples}_test_200")
                    os.makedirs(outputdir_, exist_ok=True)
                    print(outputdir_)

                    outputfile = os.path.join(outputdir_, "eval_results.json")
                    if not os.path.exists(outputfile):
                        train_data_ = train_data[:train_samples]
                        aligning_and_testing(source_model, target_model, source_tokenizer, target_tokenizer,
                                             train_data_, test_data,  outputdir_, 32,)
                    else:
                        print(f"{outputdir_} exits")


if __name__ == '__main__':
    import plac

    plac.call(aligning_per_lang_to_flant5)
