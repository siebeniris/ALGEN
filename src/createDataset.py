import torch
from torch.utils.data import Dataset, DataLoader
from inversion_utils import load_encoder_decoder_and_tokenizer, add_punctuation_token_ids, mean_pool

class InversionDataset(Dataset):
    def __init__(self, texts, tokenier, lang, encoder, device, max_length=32):
        self.texts = texts
        self.tokenizer = tokenier
        self.encoder = encoder
        self.max_length = max_length
        self.lang = lang
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

    def collate_fn(self, texts):
        tokens = add_punctuation_token_ids(texts, self.tokenizer, self.max_length, self.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # mean pooled.
        hidden_states = mean_pool(hidden_states, attention_mask)
        # labels.
        # eos_token_id=1, "."=5, pad_token_id=0
        labels = input_ids.clone()
        # Ignore padding in labels for loss computation
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # -100 is the ignore index.
        labels[labels == self.tokenizer.pad_token_id] = -100

        # true_texts = [self.tokenizer.decode(tb, skip_special_tokens=True) for tb in input_ids]
        true_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "length": token_lengths,
            "lang": self.lang,
            "text": true_texts
        }


if __name__ == '__main__':
    example_model = "google/flan-t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = [
        "This is the first example sentence.",
        "Here is another example for the dataset.",
        "The quick brown fox jumps over the lazy dog."
    ]

    encoder_decoder, tokenizer = load_encoder_decoder_and_tokenizer(example_model, device)

    encoder = encoder_decoder.encoder

    dataset = InversionDataset(texts, tokenizer, "eng-literal", encoder, device, 32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for batch in dataloader:
        hidden_states = batch["hidden_states"]
        attention_masks = batch["attention_mask"]
        input_ids = batch["input_ids"]
        print(f"h: {hidden_states.shape}, a:{attention_masks.shape}, b:{input_ids.shape}")
        print(batch)
