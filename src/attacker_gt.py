import json
import os

import numpy as np
import pandas as pd
import torch
import datasets

from data_helper import load_data_for_decoder
from decoder_finetune_trainer import DecoderFinetuneTrainer
from inversion_utils import (set_seed,
                             get_Y_tokens_from_tokenizer,
                             get_Y_embeddings_from_tokens,
                             get_mean_X,
                             load_source_encoder_and_tokenizer,
                             check_normalization)
from utils import get_device
from eval_metrics import eval_embeddings
from tqdm import tqdm


class DecoderInference:
    def __init__(self, checkpoint_path: str,
                 source_model_name: str,
                 align_train_samples: int,
                 align_test_samples: int,
                 test_dataset: str):
        """
        Initialize the inference class.
        :param checkpoint_path:
        """
        self.device = get_device()
        self.checkpoint_path = checkpoint_path

        # Loading the training arguments and best models list
        args_file_path = os.path.join(self.checkpoint_path, "training_args_and_best_models.json")

        if not os.path.exists(args_file_path):
            raise FileNotFoundError(f"File not find: {args_file_path}")

        with open(args_file_path, "r") as f:
            data = json.load(f)

        self.args = data["training_args"]
        self.best_models = data["best_models"]

        self.trainer = DecoderFinetuneTrainer(
            model_name=self.args["model_name"],
            output_dir=self.args["output_dir"],
            max_length=self.args["max_length"],
            data_folder=self.args["data_folder"],
            lang=self.args["lang"],
            train_samples=self.args["train_samples"],
            val_samples=self.args["val_samples"],
            batch_size=self.args["batch_size"],
            learning_rate=self.args["learning_rate"],
            weight_decay=self.args["weight_decay"],
            num_epochs=self.args["num_epochs"],
            wandb_run_name=self.args["wandb_run_name"],
            training_mode="test",
            checkpoint_path=None  # Do not load a checkpoint yet

        )
        self.target_model_name = self.args["model_name"]
        self.test_dataset = test_dataset
        self.source_model_name = source_model_name

        self.device = self.trainer.device
        # load the best model
        self.load_best_model()

        self.align_train_samples = align_train_samples
        self.align_test_samples = align_test_samples

        self.target_encoder = self.trainer.encoder
        self.target_tokenizer = self.trainer.tokenizer
        if self.source_model_name in ["text-embedding-3-large", "text-embedding-ada-002"]:
            print("Use the extracted embeddings...")
            self.source_encoder = None
            self.source_tokenizer = None
        else:
            self.source_encoder, self.source_tokenizer = load_source_encoder_and_tokenizer(source_model_name,
                                                                                           self.device)

        self.initialize_embeddings()

    def initialize_embeddings(self):
        if "yiyic/" in self.test_dataset:
            print(f"Loading dataset from {self.test_dataset}")
            dataset = datasets.load_dataset(self.test_dataset)
            train_texts = dataset["train"]["text"]
            val_texts = dataset["dev"]["text"]
            test_texts = dataset["test"]["text"]


        else:
            # make it ["eng", "cmn_hant"]
            print(f"loading data from {self.trainer.data_folder} and {self.test_dataset}")
            train_texts, val_texts, test_texts = load_data_for_decoder(self.trainer.data_folder, self.test_dataset)

        train_texts = train_texts[:self.align_train_samples]
        val_texts = val_texts[:self.align_test_samples]
        test_texts = test_texts[:self.align_test_samples]

        # get y tokens and true texts first, then get x tokens and texts, because no tokenization here with max_length is needed.
        Y_tokens, Y_val_tokens, Y_test_tokens = get_Y_tokens_from_tokenizer(
            train_texts,
            val_texts,
            test_texts,
            self.target_tokenizer,
            max_length=self.trainer.max_length
        )

        true_train_texts = self.target_tokenizer.batch_decode(Y_tokens["input_ids"], skip_special_tokens=True)
        true_val_texts = self.target_tokenizer.batch_decode(Y_val_tokens["input_ids"], skip_special_tokens=True)
        true_test_texts = self.target_tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True)
        print(f"train data {len(true_train_texts)}, val {len(true_val_texts)}, test {len(true_test_texts)}")

        if self.source_model_name in ["text-embedding-3-large", "text-embedding-ada-002"]:
            vector_dir = f"datasets/vectors/{self.source_model_name}"
            target_model_name_ = self.target_model_name.replace("/", "_")
            dataset_name_ = self.test_dataset.replace("/", "_")
            source_embeddings_dir = os.path.join(vector_dir, target_model_name_, dataset_name_,
                                                 f"vecs_maxlength{self.trainer.max_length}.npz")
            print(f"retrieving vectors {self.source_model_name} from {source_embeddings_dir}")
            vecs = np.load(source_embeddings_dir)
            # get the sample size accordingly
            # the embeddings are already normalized.
            X_pooled_train = torch.tensor(vecs["train"][:self.align_train_samples], dtype=torch.float32).to(self.device)
            X_pooled_val = torch.tensor(vecs["dev"][:self.align_test_samples], dtype=torch.float32).to(self.device)
            X_pooled_test = torch.tensor(vecs["test"][:self.align_test_samples], dtype=torch.float32).to(self.device)


        else:

            # get x embeddings.
            # handle texts one by one because we only get mean_pooled embeddings.
            # embeddings have the shape Bx n (batch_size, n)
            # added normalization,
            X_pooled_train = get_mean_X(true_train_texts, self.source_tokenizer, self.source_encoder, self.device,
                                        normalization=True)
            X_pooled_val = get_mean_X(true_val_texts, self.source_tokenizer, self.source_encoder, self.device,
                                      normalization=True)
            X_pooled_test = get_mean_X(true_test_texts, self.source_tokenizer, self.source_encoder, self.device,
                                       normalization=True)

        check_normalization(X_pooled_train, "X train")
        check_normalization(X_pooled_val, "X val")
        check_normalization(X_pooled_test, "X test")

        print(f"X shape train {X_pooled_train.shape}, val {X_pooled_val.shape}, test {X_pooled_test.shape}")

        Y_pooled_train = get_Y_embeddings_from_tokens(Y_tokens, self.target_encoder, normalization=True)
        Y_pooled_val = get_Y_embeddings_from_tokens(Y_val_tokens, self.target_encoder, normalization=True)
        Y_pooled_test = get_Y_embeddings_from_tokens(Y_test_tokens, self.target_encoder, normalization=True)

        check_normalization(Y_pooled_train, "Y train")
        check_normalization(Y_pooled_val, "Y val")
        check_normalization(Y_pooled_test, "Y test")

        # train data to align.
        X_aligned, T = self.mapping_X_to_Y_pooled(X_pooled_train, Y_pooled_train)
        self.source_hidden_dim, self.target_hidden_dim = T.shape

        X_val_aligned = X_pooled_val @ T
        X_test_aligned = X_pooled_test @ T

        # test alignment on X_test and Y_test
        X_Y_cos, X_Y_mseloss = eval_embeddings(X_aligned, Y_pooled_train)
        X_Y_test_cos, X_Y_test_mseloss = eval_embeddings(X_test_aligned, Y_pooled_test)
        X_Y_val_cos, X_Y_val_mseloss = eval_embeddings(X_val_aligned, Y_pooled_val)

        self.align_metrics = {
            "X_Y_COS": X_Y_cos.item(),
            "X_Y_MSEloss": X_Y_mseloss.item(),
            "X_Y_test_COS": X_Y_test_cos.item(),
            "X_Y_test_MSEloss": X_Y_test_mseloss.item(),
            "X_Y_val_COS": X_Y_val_cos.item(),
            "X_Y_val_MSEloss": X_Y_val_mseloss.item()
        }
        # print(self.align_metrics)

        # get the labels, hidden_states, and attention_mask

        self.test_data = {
            "hidden_states": X_test_aligned,
            "attention_mask": Y_test_tokens["attention_mask"],
            "texts": true_test_texts
        }

        # print(self.test_data)

    def test(self):
        self.trainer.model.eval()

        all_predictions = []
        all_references = []

        with torch.no_grad():
            inputs = {
                "hidden_states": self.test_data["hidden_states"].to(self.device),
                "attention_mask": self.test_data["attention_mask"].to(self.device)
            }

            generated_ids = self.trainer.model.generate(inputs)
            decoded_texts = self.trainer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            decoded_texts = [text.strip() for text in decoded_texts]

            all_predictions += decoded_texts
            all_references += self.test_data["texts"]

        print("decoded: ", all_predictions[:4])
        print("true: ", all_references[:4])

        test_gen_results = self.trainer.eval_texts(all_predictions, all_references)
        return test_gen_results, all_predictions, all_references

    def mapping_X_to_Y_pooled(self, X, Y):
        # mappting with normal equation.
        As = torch.linalg.pinv(X.T @ X) @ X.T @ Y
        Xs = X @ As
        return Xs, As

    def load_best_model(self):
        """
        Load the best model from the saved checkpoints.
        """
        if not self.best_models:
            raise ValueError("No best models found. Training might not have completed successfully.")

        # Load the model with the lowest validation loss
        if "yiyic" in self.checkpoint_path:
            self.best_val_loss, _, best_checkpoint_path = self.best_models[0]
        else:
            self.best_val_loss, best_checkpoint_path = self.best_models[0]
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.trainer.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Update the total number of epochs completed
        print(f"Loaded best model with val_loss={self.best_val_loss} from {best_checkpoint_path}")


def main(
        checkpoint_path="outputs/google_flan-t5-small/eng_maxlength32_train100_batch_size64_lr0.0001_wd1e-05_epochs50"
):
    test_samples = 200
    if "flan-t5-small" in checkpoint_path:
        datasets_names = ["yiyic/multiHPLT_english", "yiyic/mmarco_english", "yiyic/mmarco_german",
                          "yiyic/mmarco_spanish", "yiyic/mmarco_french"]
    else:
        datasets_names = ["yiyic/multiHPLT_english", "yiyic/mmarco_english", "yiyic/mmarco_german",
                          "yiyic/mmarco_spanish", "yiyic/mmarco_french",
                          "yiyic/mmarco_chinese", "yiyic/mmarco_vietnamese"]

    # write a loop on source model names.
    source_model_names = [
        "text-embedding-ada-002",
        "text-embedding-3-large",
        # "sentence-transformers/gtr-t5-base",
        # "intfloat/multilingual-e5-base",
        # "google/flan-t5-base",
        # "google-t5/t5-base",
        # "google/mt5-base",
        # "google-bert/bert-base-multilingual-cased",
        # "sentence-transformers/all-MiniLM-L6-v2"  # sbert
    ]

    for source_model_name in source_model_names:
        for test_data in tqdm(datasets_names):
            for train_samples in [1, 3, 5, 10, 20, 30, 40, 50, 100, 500, 1000]:
                source_model_name_ = source_model_name.replace("/", "_")
                test_dataset_ = test_data.replace("/", "_")
                output_dir = os.path.join(checkpoint_path,
                                          f"attack_{test_dataset_}_{source_model_name_}_train{train_samples}")

                # if not os.path.exists(output_dir):
                print(f"attacking embeddings from {source_model_name} with {train_samples} train samples")
                decoderInference = DecoderInference(checkpoint_path, source_model_name,
                                                    train_samples, test_samples, test_data)

                test_results, preds, references = decoderInference.test()
                # TODO: add translation here.
                # translate the decoded into fine-tuned language.
                # extract language from checkpoints and dataset name
                # use easy nmt

                df_preds_ref = pd.DataFrame({"predictions": preds, "reference": references})

                results_dict = {
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                    "source_model": source_model_name,
                    "source_dim": decoderInference.source_hidden_dim,
                    "target_dim": decoderInference.target_hidden_dim,
                    "test_results": test_results,
                    "loss": decoderInference.align_metrics
                }
                print(results_dict)

                print(f"writing the results to {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                df_preds_ref.to_csv(os.path.join(output_dir, "results_texts.csv"))
                with open(os.path.join(output_dir, "results.json"), "w") as f:
                    json.dump(results_dict, f)
                print("*" * 40)


if __name__ == '__main__':
    set_seed(42)
    import plac

    plac.call(main)
