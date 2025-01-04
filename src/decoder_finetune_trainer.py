import json
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import evaluate
import wandb

from decoder_finetune import DecoderFinetuneModel
from createDataset import InversionDataset
from data_helper import load_data_for_decoder
from eval_metrics import get_rouge_scores

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DecoderFinetuneTrainer:
    def __init__(self,
                 model_name: str,
                 output_dir: str,
                 max_length: int = 32,
                 data_folder: str = "datasets/finetuning_decoder",
                 lang: str = "eng",
                 train_samples: int = 100,
                 val_samples: int = 200,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,  # (T5)
                 weight_decay: float = 1e-5,
                 num_epochs: int = 50,
                 wandb_run_name: str = "decoder_finetuning",
                 checkpoint_path: str = None,
                 training_mode: str = None
                 ):

        self.args = {
            "model_name": model_name,
            "output_dir": output_dir,
            "max_length": max_length,
            "data_folder": data_folder,
            "lang": lang,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "wandb_run_name": wandb_run_name,
            "checkpoint_path": checkpoint_path
        }

        # initialization code
        self.max_length = max_length
        self.model = DecoderFinetuneModel(model_name, self.max_length)
        self.data_folder = data_folder
        self.lang = lang
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.device = self.model.device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.tokenizer = self.model.tokenizer
        self.encoder = self.model.encoder_decoder.encoder
        # initialize the resources
        self.initialize_resources()

        self.output_dir = os.path.join(output_dir,
                                       model_name.replace("/", "_"),
                                       f"{lang}_maxlength{max_length}_train{train_samples}_batch_size{batch_size}_lr{learning_rate}_wd{weight_decay}_epochs{num_epochs}")

        os.makedirs(self.output_dir, exist_ok=True)

        # continue training
        # Check if training_args_and_best_models.json exists

        if training_mode == "continue":
            args_file_path = os.path.join(self.output_dir, "training_args_and_best_models.json")
            if os.path.exists(args_file_path):
                print(f"Found existing training arguments and best models at {args_file_path}. Loading...")
                with open(args_file_path, "r") as f:
                    data = json.load(f)
                if self.args == data["training_args"]:
                    print(f"Training arguments match. Loading the best model...")
                    self.best_models = data["best_models"]
                    self.load_best_model()  # Load the best model
                else:
                    print("Training arguments do not match. Finetuning decoder fresh.")
                    self.best_models = []
                    self.best_val_loss = float("inf")
        else:
            print(f"Initializing best models and best val loss...")
            self.best_models = []
            self.best_val_loss = float("inf")

        if training_mode != "test":

            self.wandb_project = model_name.replace("/", "_") + "_" + wandb_run_name
            if self.wandb_project:
                wandb.init(project=self.wandb_project, name=wandb_run_name,
                           config={
                               "model_name": model_name,
                               "max_length": max_length,
                               "data_folder": data_folder,
                               "lang": lang,
                               "train_samples": train_samples,
                               "val_samples": val_samples,
                               "batch_size": batch_size,
                               "learning_rate": learning_rate,
                               "num_epochs": num_epochs,
                           })

        if checkpoint_path:
            self.load_model_from_checkpoint(checkpoint_path)

    def initialize_resources(self):
        train_texts, val_texts, _ = load_data_for_decoder(self.data_folder, self.lang)
        train_texts = train_texts[:self.train_samples]
        val_texts = val_texts[:self.val_samples]
        self.train_dataset = InversionDataset(train_texts, self.tokenizer, self.lang, self.encoder,
                                              self.device, self.max_length)
        self.val_dataset = InversionDataset(val_texts, self.tokenizer, self.lang, self.encoder,
                                            self.device, self.max_length)

        # load the train and val dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=self.train_dataset.collate_fn,
                                       num_workers=0
                                       )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False,
                                     collate_fn=self.val_dataset.collate_fn,
                                     num_workers=0
                                     )
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.rouge_metric = get_rouge_scores
        self.bleu_metric = evaluate.load("sacrebleu")

    def train(self):
        """
        Train the model
        """
        try:
            # check if start_epoch exists, it means a fine-tuned model is loaded.
            start_epoch = getattr(self, "start_epoch", 0)
            print(f"the starting epoch for training is {start_epoch}")
            for epoch in range(start_epoch, self.num_epochs):
                self.model.train()

                epoch_loss = 0.0
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

                for batch in progress_bar:
                    inputs = {
                        "hidden_states": batch["hidden_states"].to(self.device),
                        "attention_mask": batch["attention_mask"].to(self.device),
                        "labels": batch["labels"].to(self.device),
                    }
                    outputs = self.model(inputs)
                    loss = outputs.loss

                    # clears the old gradients
                    self.optimizer.zero_grad()
                    # computes the derivative of the loss w.r.t the parameters using backpropagation
                    loss.backward()
                    # take a step based on the gradients of the parameters
                    self.optimizer.step()

                    # update the progress
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": loss.item()})

                avg_epoch_loss = epoch_loss / len(self.train_loader)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss {avg_epoch_loss}")

                val_loss, gen_results = self.validate()
                print(val_loss)
                print(gen_results)

                # log to wandb
                if wandb.run:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": avg_epoch_loss,
                        "val_loss": val_loss,
                        **gen_results
                    })

                # save the best model
                self.save_best_model(val_loss, gen_results, epoch + 1)
        finally:
            if wandb.run:
                wandb.finish()

    def validate(self):
        """
        Validate the model on the validation dataset.
        """
        self.model.eval()

        val_loss = 0.0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = {
                    "hidden_states": batch["hidden_states"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["labels"].to(self.device)
                }
                outputs = self.model(inputs)
                loss = outputs.loss
                val_loss += loss.item()

                # generated text.
                generated_ids = self.model.generate(inputs)
                decoded_text = self.model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                decoded_text = [text.strip() for text in decoded_text]
                # print(f"decoded text:", decoded_text)
                # print(f"true text:", batch["text"])

                all_predictions += decoded_text
                all_references += batch["text"]

        print(f"decoded text:", all_predictions[:4])
        print(f"true text:", all_references[:4])

        gen_results = self.eval_texts(all_predictions, all_references)
        return val_loss, gen_results

    def eval_texts(self, predictions, references):
        bleu_scores = self.bleu_metric.compute(predictions=predictions, references=references)
        rouge_score = self.rouge_metric(predictions, references)
        print(bleu_scores)
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

    def save_best_model(self, val_loss, val_results, epoch):
        """
        Save the best model based on validation loss.
        Keep at most 3 models.
        :param val_loss:
        :param epoch:
        :return:
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_results": val_results
                },
                checkpoint_path,
            )

            # Add to the best models list
            self.best_models.append((val_loss, checkpoint_path))
            self.best_models.sort(key=lambda x: x[0])  # Sort by val_loss

            # Keep only the top 2 models
            if len(self.best_models) > 2:
                # pop the last element, which has the biggest val_loss
                _, oldest_checkpoint = self.best_models.pop()
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)

            # save all arguments and the best models list to a JSON file.
            args_file_path = os.path.join(self.output_dir, "training_args_and_best_models.json")
            with open(args_file_path, "w") as f:
                json.dump({
                    "training_args": self.args,
                    "best_models": self.best_models,
                }, f, indent=4)

            print(f"Saved new best model with val_loss={val_loss} at {checkpoint_path}")
            print(f"Saved training arguments and best models at {args_file_path}")
        else:
            print(f"Validation loss did not improve ({val_loss} >= {self.best_val_loss}). Skipping model save.")

    def load_model_from_checkpoint(self, checkpoint_path):
        """
                Load the model and optimizer states from a custom checkpoint path.

                Args:
                    checkpoint_path (str): Path to the checkpoint file.
                """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Optionally, load other information (e.g., epoch, val_loss)
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Loaded model from {checkpoint_path} (epoch={self.start_epoch}, val_loss={val_loss})")

    def load_best_model(self):
        """
        Load the best model from the saved checkpoints.
        """
        if not self.best_models:
            raise ValueError("No best models found. Training might not have completed successfully.")

        # Load the model with the lowest validation loss
        self.best_val_loss, best_checkpoint_path = self.best_models[0]
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Update the total number of epochs completed
        self.start_epoch = checkpoint.get("epoch", 0) + 1  # Start from the next epoch
        print(f"Loaded best model with val_loss={self.best_val_loss} from {best_checkpoint_path}")
        print(f"Resuming training from epoch {self.start_epoch}")
