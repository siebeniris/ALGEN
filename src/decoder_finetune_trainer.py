import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import evaluate

from finetune_decoder import DecoderFinetuneModel
from createDataset import InversionDataset
from utils import get_device
from data_helper import load_data_for_decoder
from eval_metrics import get_rouge_scores


class DecoderFinetuneTrainer:
    def __init__(self,
                 model_name: str,
                 output_dir: str,
                 max_length: int = 32,
                 data_folder: str = "eng-literal",
                 train_samples: int = 100,
                 val_samples: int = 10,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,  # (T5)
                 num_epochs: int = 50,
                 save_every: int = 2,  # save the best?
                 ):

        self.max_length = max_length
        self.model = DecoderFinetuneModel(model_name, self.max_length)
        self.data_folder = data_folder

        if len(data_folder) == 3:
            self.lang = data_folder
        else:
            self.lang = data_folder.split("-")[0]

        self.train_samples = train_samples
        self.val_samples = val_samples

        self.device = self.model.device
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_every = save_every

        self.tokenizer = self.model.tokenizer
        self.encoder = self.model.encoder_decoder.encoder
        # initialize the resources
        self.initialize_resources()

    def initialize_resources(self):
        os.makedirs(self.output_dir, exist_ok=True)

        train_texts, val_texts = load_data_for_decoder(self.data_folder)
        train_texts = train_texts[:self.train_samples]
        val_texts = val_texts[:self.val_samples]
        self.train_dataset = InversionDataset(train_texts, self.tokenizer, self.lang, self.encoder,
                                              self.device, self.max_length)
        self.val_dataset = InversionDataset(val_texts, self.tokenizer, self.lang, self.encoder,
                                            self.device, self.max_length)

        # load the train and val dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=self.train_dataset.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False,
                                     collate_fn=self.val_dataset.collate_fn)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.rouge_metric = get_rouge_scores
        self.bleu_metric = evaluate.load("bleu")

    def train(self):
        """
        Train the model
        """
        for epoch in range(self.num_epochs):
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

            # validation

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
                print(f"decoded text:", decoded_text)
                print(f"true text:", batch["text"])

                all_predictions += decoded_text
                all_references += batch["text"]

        gen_results = self.eval_texts(all_predictions, all_references)
        return val_loss, gen_results

    def eval_texts(self, predictions, references):
        bleu_scores = np.array(
            [
                self.bleu_metric.compute(predictions=[p], references=[r])["bleu"]
                for p, r in zip(predictions, references)
            ]
        )
        rouge_score = self.rouge_metric(predictions, references)
        rouge_score_dict = {k: np.mean(v) for k, v in rouge_score.items()}
        exact_matches = np.array(predictions) == np.array(references)
        gen_metrics = {
            "bleu": bleu_scores.mean(),
            "rougeL": rouge_score_dict["rougeL_f"],
            "rouge1": rouge_score_dict["rouge1_f"],
            "rouge2": rouge_score_dict["rouge2_f"],
            "exact_match": sum(exact_matches) / len(exact_matches)
        }
        # oracle : 70 rougeL
        return gen_metrics



if __name__ == '__main__':
    example_model = "google/flan-t5-small"
    device = get_device()
    outputdir = f"outputs/{example_model.replace("/", "-")}"

    trainer = DecoderFinetuneTrainer(example_model, outputdir)
    trainer.train()
