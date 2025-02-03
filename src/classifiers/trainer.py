import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from transformers import AutoTokenizer, AutoModel, default_data_collator
from datasets import load_dataset
from data_helper import EmbeddingDataset, preprocess_data, extract_embeddings, save_embeddings, load_embeddings
from eval_utils import eval_classification
from model import Classifier
from tqdm import tqdm


# Train function
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        embeddings, labels = batch
        embeddings, labels = embeddings.to(device), labels.to(device)
        outputs = model(embeddings)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# Evaluate function
def evaluation_step(model, dataloader, task, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            # print(outputs)
            prob_scores = torch.softmax(outputs, dim=-1)

            predictions.extend(prob_scores.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return eval_classification(true_labels, predictions, task)


def fine_tune(dataset_name, task_name, num_labels, model_name,
              batch_size=128,
              defense_method="NoDefense",
              output_dir="outputs/classifiers/",
              epochs=6, learning_rate=3e-4):

    assert task_name in ["sentiment", "nli"]
    assert dataset_name in ["yiyic/snli_ds", "yiyic/sst2_ds", "yiyic/s140_ds"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device {device}")

    model_name_ = model_name.replace("/", "__")
    dataset_name_ = dataset_name.replace("/", "__")

    output_dir = os.path.join(output_dir, model_name_, dataset_name_, defense_method)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "datasets/"
    embedding_dir = os.path.join(data_dir, f"{model_name_}_{dataset_name_}_{defense_method}")
    embedding_path = os.path.join(embedding_dir, "train_embeddings.pt")

    os.makedirs(embedding_dir, exist_ok=True)

    print(f"loading dataset... {dataset_name}")
    dataset = load_dataset(dataset_name)

    best_acc = - np.inf

    if model_name in ["google-t5/t5-base", "google/mt5-base", "google-bert/bert-base-multilingual-cased"]:
        embedding_dim = 768
        num_labels = int(num_labels)

        if not os.path.exists(embedding_path):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            encoder = AutoModel.from_pretrained(model_name)
            encoder = encoder.to(device)

            tokenized_dataset = preprocess_data(dataset, tokenizer, task_name)

            # prepare pytorch datasets.
            train_dataset = tokenized_dataset["train"]
            dev_dataset = tokenized_dataset["dev"]
            test_dataset = tokenized_dataset["test"]

            # dataloader
            print(f"creating dataloaders")
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
            dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=default_data_collator)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_data_collator)

            # embeddings and labels.
            print(f"creating embeddings and labels.")
            train_embeddings, train_labels = extract_embeddings(encoder, train_dataloader, device=device)
            dev_embeddings, dev_labels = extract_embeddings(encoder, dev_dataloader, device=device)
            test_embeddings, test_labels = extract_embeddings(encoder, test_dataloader, device=device)

            print(f"saving embeddings and labels to {data_dir}")
            save_embeddings(train_embeddings, train_labels, embedding_dir, "train")
            save_embeddings(dev_embeddings, dev_labels, embedding_dir, "dev")
            save_embeddings(test_embeddings, test_labels, embedding_dir, "test")


        else:
            print(f"Loading embeddings from {embedding_dir}")
            train_embeddings, train_labels = load_embeddings(embedding_dir, "train")
            dev_embeddings, dev_labels = load_embeddings(embedding_dir, "dev")
            test_embeddings, test_labels = load_embeddings(embedding_dir, "test")

        # create pytorch dataset for embeddings
        print(f"creating embeddings dataset.")
        train_embedding_dataset = EmbeddingDataset(train_embeddings, train_labels)
        dev_embedding_dataset = EmbeddingDataset(dev_embeddings, dev_labels)
        test_embedding_dataset = EmbeddingDataset(test_embeddings, test_labels)

        # data loaders.
        print(f"creating embeddings dataloaders.")
        train_embedding_dataloader = DataLoader(train_embedding_dataset, batch_size=batch_size)
        dev_embedding_dataloader = DataLoader(dev_embedding_dataset, batch_size=batch_size)
        test_embedding_dataloader = DataLoader(test_embedding_dataset, batch_size=batch_size)

        # create dataloader for embeddings for training.

        print(f"device {device}, embedding dim {embedding_dim} type {type(embedding_dim)}  num labels {num_labels}")

        classifier = Classifier(embedding_dim, num_labels).to(device)

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)

        for epoch in tqdm(range(epochs)):
            train_loss = train(classifier, train_embedding_dataloader, optimizer, device)
            if task_name == "nli":
                dev_acc, dev_f1, dev_auc = evaluation_step(classifier, dev_embedding_dataloader, "multiclass", device)
            else:
                dev_acc, dev_f1, dev_auc = evaluation_step(classifier, dev_embedding_dataloader, "binary", device)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")
            print(f"Dev result: acc: {dev_acc}")

            if dev_acc > best_acc:
                best_acc = dev_acc

                print("testing ...")

                if task_name == "nli":
                    test_acc, test_f1, test_auc = evaluation_step(classifier, test_embedding_dataloader, "multiclass",
                                                                  device)
                else:
                    test_acc, test_f1, test_auc = evaluation_step(classifier, test_embedding_dataloader, "binary",
                                                                  device)

                test_results = {
                    "epoch": epoch,
                    "dev_acc": best_acc,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_roc": test_auc
                }

                with open(os.path.join(output_dir, f"epoch_{epoch}_results.json"), "w") as f:
                    json.dump(test_results, f)


if __name__ == '__main__':
    import plac

    plac.call(fine_tune)
