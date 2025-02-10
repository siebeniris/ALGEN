from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm


def mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def preprocess_data(dataset, tokenizer, task_name, max_length=32):
    def tokenize_function(examples):
        if task_name == "sentiment":
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length,
                             return_tensors="pt")
        elif task_name == "nli":
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")
        else:
            raise ValueError("Task not supported.")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def extract_embeddings(model, dataloader, device: torch.device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # more efficient with fp16
            # with torch.autocast(device_type=device.type):
            # probably the problem for nans.?
            if model.config.is_encoder_decoder:
                    outputs = model.encoder(input_ids, attention_mask=attention_mask)
            else:
                    outputs = model(input_ids, attention_mask=attention_mask)

            last_hidden_state_ = outputs.last_hidden_state
            # mean pooled embeddings
            mean_pooled_embeddings = mean_pool(last_hidden_state_, attention_mask)
            # normalized
            normed_embeddings = torch.nn.functional.normalize(mean_pooled_embeddings, p=2, dim=1)

            embeddings.append(normed_embeddings.cpu())
            labels.append(batch["labels"].cpu())
    # new dimension. [samples, token_length, hidden_dim]
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels



def extract_random_embeddings(dataloader, device:torch.device, embeddings_dim=768):
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # print(batch)
            batch_size = batch["input_ids"].shape[0]
            random_embeddings = torch.randn(batch_size, embeddings_dim, device=device)

            normed_embeddings = torch.nn.functional.normalize(random_embeddings, p=2, dim=1)

            embeddings.append(normed_embeddings.cpu())
            labels.append(batch["labels"].cpu())
    # new dimension. [samples, token_length, hidden_dim]
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels



def save_embeddings(embeddings, labels, save_dir, data_split):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(embeddings, os.path.join(save_dir, f"{data_split}_embeddings.pt"))
    torch.save(labels, os.path.join(save_dir, f"{data_split}_labels.pt"))


def load_embeddings(save_dir, data_split):
    print(f"loading {data_split} embeddings from {save_dir}")
    embeddings = torch.load(os.path.join(save_dir, f"{data_split}_embeddings.pt"))
    labels = torch.load(os.path.join(save_dir, f"{data_split}_labels.pt"))
    return embeddings, labels


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
