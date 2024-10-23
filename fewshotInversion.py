import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from sklearn.metrics import pairwise
import evaluate
import transformers
from transformers import default_data_collator


class TextEmbeddingDataset(Dataset):
    def __init__(self, texts, emb_g, emb_s):
        """
        :param texts: List of original input texts.
        :param emb_g: Embeddings from encoder g.
        :param emb_s: Embeddings from encoder s.
        """
        self.texts = texts
        self.emb_g = emb_g
        self.emb_s = emb_s

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'emb_g': torch.tensor(self.emb_g[idx], dtype=torch.float32),
            'emb_s': torch.tensor(self.emb_s[idx], dtype=torch.float32),
        }

class AlignmentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AlignmentModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, emb_s):
        return self.linear(emb_s)

class CosineSimilarityLoss(nn.Module):
    def forward(self, aligned_emb_s, emb_g):
        cos_sim = nn.functional.cosine_similarity(aligned_emb_s, emb_g, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss


def load_encoder_decoder(
    model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )

def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer


def load_embedder_and_tokenizer(name:str, **kwargs):
    model_kwargs = {
        "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": False,  # True output hidden states, for embedding last and first .
    }
    # TODO: check the configurations for commercial models

    if name=="me5":
        model = transformers.AutoModel.from_pretrained("intfloat/multilingual-e5-base", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = transformers.AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    return model, tokenizer


def get_encoder_embeddings(model, tokenizer, input_texts):
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in input_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # Use mean pooling
            embeddings.append(embedding)
            print(embedding.shape)

    return torch.tensor(embeddings)


training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        logging_dir="./logs",
        logging_steps=10,
         remove_unused_columns=False, #very important.
    use_mps_device=True
    )


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        emb_s = inputs['emb_s']
        emb_g = inputs['emb_g']

        aligned_emb_s = model(emb_s)
        loss_fn = CosineSimilarityLoss()
        loss = loss_fn(aligned_emb_s, emb_g)

        return (loss, aligned_emb_s) if return_outputs else loss



if __name__ == '__main__':
    texts = [
        "This is a test sentence.",
        "I love natural language processing.",
        "Transformers are amazing models!"
    ]

    # Load embeddings from encoder g and encoder s
    encoder_model_name = "intfloat/multilingual-e5-small"
    encoder_decoder_model_name = "google-t5/t5-small"
    encoder, encoder_tokenizer = load_embedder_and_tokenizer("me5")
    encoder_decoder = load_encoder_decoder(encoder_decoder_model_name)
    encoder_decoder_tokenizer = load_tokenizer(encoder_decoder_model_name, max_length=128)

    emb_g = get_encoder_embeddings(encoder_decoder.encoder, encoder_decoder_tokenizer, texts)
    emb_s = get_encoder_embeddings(encoder, encoder_tokenizer, texts)
    dataset = TextEmbeddingDataset(texts, emb_g, emb_s)

    input_dim = emb_s.shape[1]
    output_dim = emb_g.shape[1]
    model = AlignmentModel(input_dim, output_dim)

    # Create custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_s = emb_s.to(device)
    model.to(device)
    aligned_emb_s = model(emb_s)



    aligned_emb = aligned_emb_s.unsqueeze(0)
    print(aligned_emb.shape)
    # Create a named tuple to pass encoder outputs manually
    from transformers.modeling_outputs import BaseModelOutput

    encoder_outputs = BaseModelOutput(last_hidden_state=aligned_emb)
    # print(encoder_outputs.shape)

    # Provide initial decoder input IDs (start token)
    decoder_input_ids = torch.tensor([[encoder_decoder.config.decoder_start_token_id]])

    # Generate text based on aligned embeddings
    decoder_outputs = encoder_decoder.generate(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
        max_length=50,  # Limit the generation length
        num_beams=5  # Beam search for better results (optional)
    )

    # Decode the generated tokens into text
    generated_text = encoder_decoder_tokenizer.decode(decoder_outputs[0], skip_special_tokens=True)
    print(generated_text)

    # no output,

