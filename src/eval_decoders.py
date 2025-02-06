import json
import os

import pandas as pd
import torch
import datasets

from data_helper import load_data_for_decoder
from decoder_finetune_trainer import DecoderFinetuneTrainer
from inversion_utils import (set_seed,
                             get_Y_tokens_from_tokenizer,
                             get_Y_embeddings_from_tokens,
                             load_source_encoder_and_tokenizer,
                             check_normalization)
from utils import get_device
from tqdm import tqdm


set_seed(42)

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
        _, _, Y_test_tokens = get_Y_tokens_from_tokenizer(
            train_texts,
            val_texts,
            test_texts,
            self.target_tokenizer,
            max_length=self.trainer.max_length
        )

        # true_train_texts = self.target_tokenizer.batch_decode(Y_tokens["input_ids"], skip_special_tokens=True)
        # true_val_texts = self.target_tokenizer.batch_decode(Y_val_tokens["input_ids"], skip_special_tokens=True)
        true_test_texts = self.target_tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True)
        print(f"test {len(true_test_texts)}")

        Y_pooled_test = get_Y_embeddings_from_tokens(Y_test_tokens, self.target_encoder, normalization=True)
        check_normalization(Y_pooled_test, "Y test")

        self.test_data = {
            "hidden_states": Y_pooled_test,
            "attention_mask": Y_test_tokens["attention_mask"],
            "texts": true_test_texts
        }


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
        checkpoint_path="outputs/google_flan-t5-small/yiyic_multiHPLT_english_maxlength32_train150000_batch_size128_lr0.0001_wd0.0001_epochs100"
):
    test_samples = 200
    if "flan-t5-small" in checkpoint_path:
        datasets_names = ["yiyic/multiHPLT_english", "yiyic/mmarco_english", "yiyic/mmarco_german",
                          "yiyic/mmarco_spanish", "yiyic/mmarco_french"]

        source_model_names = ["google/flan-t5-small"]

        for source_model_name in source_model_names:
            for test_data in tqdm(datasets_names):
                    source_model_name_ = source_model_name.replace("/", "_")
                    test_dataset_ = test_data.replace("/", "_")
                    output_dir = os.path.join(checkpoint_path,
                                              f"decoder_eval_{test_dataset_}_{source_model_name_}")

                    # if not os.path.exists(output_dir):
                    print(f"attacking embeddings from {source_model_name} with")
                    decoderInference = DecoderInference(checkpoint_path, source_model_name,
                                                        100, test_samples, test_data)

                    test_results, preds, references = decoderInference.test()
                    # TODO: add translation here.
                    # translate the decoded into fine-tuned language.
                    # extract language from checkpoints and dataset name
                    # use easy nmt

                    df_preds_ref = pd.DataFrame({"predictions": preds, "reference": references})

                    results_dict = {
                        "test_samples": test_samples,
                        "source_model": source_model_name,
                        # "source_dim": decoderInference.source_hidden_dim,
                        # "target_dim": decoderInference.target_hidden_dim,
                        "test_results": test_results,
                        # "loss": decoderInference.align_metrics
                    }
                    print(results_dict)

                    print(f"writing the results to {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    df_preds_ref.to_csv(os.path.join(output_dir, "results_texts.csv"))
                    with open(os.path.join(output_dir, "results.json"), "w") as f:
                        json.dump(results_dict, f)
                    print("*" * 40)


if __name__ == '__main__':
    import plac
    plac.call(main)