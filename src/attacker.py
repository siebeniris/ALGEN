import json
import os

import torch
import datasets

from data_helper import load_data_for_decoder
from decoder_finetune_trainer import DecoderFinetuneTrainer
from inversion_utils import set_seed, mean_pool, load_source_encoder_and_tokenizer, get_embeddings_from_encoders
from utils import get_device
from eval_metrics import eval_embeddings


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
        self.test_dataset = test_dataset


        self.device = self.trainer.device
        # load the best model
        self.load_best_model()

        self.align_train_samples = align_train_samples
        self.align_test_samples = align_test_samples

        self.target_encoder = self.trainer.encoder
        self.target_tokenizer = self.trainer.tokenizer
        self.source_encoder, self.source_tokenizer = load_source_encoder_and_tokenizer(source_model_name, self.device)

        self.initialize_embeddings()

    def initialize_embeddings(self):
        if "yiyic/" in self.test_dataset:
            print(f"Loading dataset from {self.test_dataset}")
            dataset = datasets.load_dataset(self.test_dataset)
            train_texts = dataset["train"]["text"]
            test_texts = dataset["test"]["text"]

        else:
            # make it ["eng", "cmn_hant"]
            print(f"loading data from {self.trainer.data_folder} and {self.test_dataset}")
            train_texts, _, test_texts = load_data_for_decoder(self.trainer.data_folder, self.test_dataset)

        train_texts = train_texts[:self.align_train_samples]
        test_texts = test_texts[:self.align_test_samples]

        X, Y, X_test, Y_test, X_tokens, Y_tokens, X_test_tokens, Y_test_tokens \
            = get_embeddings_from_encoders(
            train_texts, test_texts,
            self.source_encoder, self.target_encoder,
            self.source_tokenizer, self.target_tokenizer,
            max_length=self.trainer.max_length
        )

        self.source_hidden_dim = X.shape[-1]
        self.target_hidden_dim = Y.shape[-1]

        X_attention_mask = X_tokens["attention_mask"]
        Y_attention_mask = Y_tokens["attention_mask"]
        X_test_attention_mask = X_test_tokens["attention_mask"]
        Y_test_attention_mask = X_test_tokens["attention_mask"]  # what we need for inference.

        # mean pooled embeddings.
        X_pooled = mean_pool(X, X_attention_mask)
        Y_pooled = mean_pool(Y, Y_attention_mask)
        X_test_pooled = mean_pool(X_test, X_test_attention_mask)
        Y_test_pooled = mean_pool(Y_test, Y_test_attention_mask)

        # train data to align.
        X_aligned, T = self.mapping_X_to_Y_pooled(X_pooled, Y_pooled)
        X_test_aligned = X_test_pooled @ T

        # test alignment on X_test and Y_test
        X_Y_cos, X_Y_mseloss = eval_embeddings(X_aligned, Y_pooled)
        X_Y_test_cos, X_Y_test_mseloss = eval_embeddings(X_test_aligned, Y_test_pooled)
        self.align_metrics = {
            "X_Y_COS": X_Y_cos.item(),
            "X_Y_MSEloss": X_Y_mseloss.item(),
            "X_Y_test_COS": X_Y_test_cos.item(),
            "X_Y_test_MSEloss": X_Y_test_mseloss.item()
        }
        # print(self.align_metrics)

        # get the labels, hidden_states, and attention_mask
        true_texts = self.target_tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True)

        self.test_data = {
            "hidden_states": X_test_aligned,
            "attention_mask": Y_test_attention_mask,
            "texts": true_texts
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
        return test_gen_results

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
        self.best_val_loss, _,  best_checkpoint_path = self.best_models[0]
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.trainer.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Update the total number of epochs completed
        print(f"Loaded best model with val_loss={self.best_val_loss} from {best_checkpoint_path}")


def main(
        test_data,
        checkpoint_path="outputs/google_flan-t5-small/eng_maxlength32_train100_batch_size64_lr0.0001_wd1e-05_epochs50"

):
    test_samples = 200
    # write a loop on source model names.
    source_model_names = [
        "sentence-transformers/gtr-t5-base",
        "intfloat/multilingual-e5-base",
        "google/flan-t5-base",
        "google-t5/t5-base",
        "google/mt5-base",
        "google-bert/bert-base-multilingual-cased"
    ]

    for source_model_name in source_model_names:
        for train_samples in [3, 5, 10, 15, 20, 30, 40, 50, 100, 500, 1000]:
            print(f"attacking embeddings from {source_model_name} with {train_samples} train samples")
            decoderInference = DecoderInference(checkpoint_path, source_model_name,
                                                train_samples, test_samples, test_data)

            test_results = decoderInference.test()

            results_dict = {
                "train_samples": train_samples,
                "test_samples": test_samples,
                "source_model": source_model_name,
                "source_hidden_dim": decoderInference.source_hidden_dim,
                "target_hidden_dim": decoderInference.target_hidden_dim,
                "test_results": test_results,
                "loss": decoderInference.align_metrics
            }
            print(results_dict)
            source_model_name_ = source_model_name.replace("/", "_")
            print(f"writing the results to {checkpoint_path}")
            test_dataset= test_data.replcae("/", "_")
            with open(os.path.join(checkpoint_path, f"test_results_{test_dataset}_{source_model_name_}_train{train_samples}_.json"),
                      "w") as f:
                json.dump(results_dict, f)
            print("*"*40)


if __name__ == '__main__':
    set_seed(42)
    import plac

    plac.call(main)
