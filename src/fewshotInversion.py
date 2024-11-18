import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re
import Levenshtein
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm, trange
from transformers.modeling_outputs import BaseModelOutput

from alignment_models import LinearAligner

device = torch.device("cpu" if torch.has_mps else "cpu")
print(f"Set to device {device}")

def load_models_and_tokenizers():
    # load the models and tokenizer
    gtr_model = SentenceTransformer("sentence-transformers/gtr-t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    gtr_model = gtr_model.to(device)
    t5_model = t5_model.to(device)
    return t5_tokenizer, t5_model, gtr_model


def get_t5_embeddings(sentences, t5_tokenizer,t5_model):

    inputs = t5_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        encoder_outputs = t5_model.encoder(input_ideds=inputs.input_ids, attention_mask=inputs.attention_mask)
    return encoder_outputs.last_hidden_state[:, 0, :]  # Get [CLS] token embedding for simplicity


def train_alignment(sentences, t5_tokenizer, t5_model, gtr_model):

    with torch.no_grad():
        gtr_embeddings = gtr_model.encode(sentences, convert_to_tensor=True)
        t5_embeddings = get_t5_embeddings(sentences, t5_tokenizer,t5_model)

    alignment_model = AlignmentModel(gtr_embeddings.size(1), t5_embeddings.size(1))
    alignment_model = alignment_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(alignment_model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        alignment_model.train()
        optimizer.zero_grad()

        # Forward pass
        aligned_embeddings = alignment_model(gtr_embeddings.to(device))
        loss = criterion(aligned_embeddings, t5_embeddings.to(device))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return alignment_model


def get_closest_tokens(aligned_embeddings, embedding_layer, top_k=1):
    """Retrieve closest tokens for each aligned embedding in T5's vocabulary."""
    vocab_embeddings = embedding_layer.weight  # Vocabulary embeddings from T5's embedding layer
    closest_tokens = []
    for embedding in aligned_embeddings:
        embedding = embedding.to(device)
        # Calculate cosine similarity between the embedding and all vocab embeddings
        similarities = F.cosine_similarity(embedding.unsqueeze(0), vocab_embeddings, dim=1)
        topk_indices = torch.topk(similarities, top_k).indices
        closest_tokens.append(topk_indices[0].item())  # Get the closest token index
    return closest_tokens


def safe_decode(token_ids, tokenizer):
    """Decode token IDs, handling out-of-range IDs by replacing them with <unk>."""
    valid_token_ids = [t if 0 <= t < tokenizer.vocab_size else tokenizer.unk_token_id for t in token_ids]
    return tokenizer.decode(valid_token_ids, skip_special_tokens=True)


def decode_embeddings_with_t5(aligned_embeddings, t5_model, t5_tokenizer):
    decoded_sentences = []
    for embedding in aligned_embeddings:
        # Find the closest token to use as initial input
        closest_token_id = get_closest_tokens(embedding.unsqueeze(0), t5_tokenizer, t5_model.shared)[0]

        # Generate using the closest token as the initial input
        input_ids = torch.tensor([[closest_token_id]]).to(device)
        outputs = t5_model.generate(input_ids=input_ids, max_length=50, num_beams=5)

        # Decode the generated output tokens to text
        # decoded_sentence = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_sentence = safe_decode(outputs[0], t5_tokenizer)
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences


def eval_alignment(test_sentences, alignment_model, gtr_model, t5_model, t5_tokenizer):

    with torch.no_grad():
        gtr_embeddings = gtr_model.encode(test_sentences, convert_to_tensor=True)
        aligned_embeddings = alignment_model(gtr_embeddings)
        decoded_sentences = decode_embeddings_with_t5(aligned_embeddings, t5_model, t5_tokenizer)

    # Display results
    for i, (original, decoded) in enumerate(zip(test_sentences, decoded_sentences)):
        print(f"Original: {original}")
        print(f"Decoded: {decoded}")
        print("-" * 50)

def main():
    sentences = [
        "The sky is blue.", "I love pizza.", "The cat sat on the mat.",
        "This is a test sentence.", "Artificial intelligence is fascinating.",
    ]
    t5_tokenizer, t5_model, gtr_model = load_models_and_tokenizers()

    # training
    alignment_model = train_alignment(sentences, t5_tokenizer, t5_model, gtr_model)

    # get another dataset for sentences
    eval_alignment(sentences, alignment_model, gtr_model, t5_tokenizer, t5_model)



if __name__ == '__main__':
    main()