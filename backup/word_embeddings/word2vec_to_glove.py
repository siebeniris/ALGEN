import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("loading word2vec model ")
word2vec_model = KeyedVectors.load_word2vec_format("vectors/GoogleNews-vectors-negative300.bin", binary=True)
word2vec_vocab = set(word2vec_model.key_to_index.keys())

print("loading glove model..")
glove_model = {}
with open("vectors/glove.42B.300d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        glove_model[word] = vector

common_vocab = set(word2vec_model.key_to_index.keys()).intersection(glove_model.keys())

print("the 10 overlapping keys")
print(list(common_vocab)[:10])

word2vec_matrix = np.array([word2vec_model[word] for word in common_vocab])
glove_matrix = np.array([glove_model[word] for word in common_vocab])

print("implementing orthogonal procrustes")
R, _ = orthogonal_procrustes(word2vec_matrix, glove_matrix)

aligned_word2vec = {word: word2vec_model[word].dot(R) for word in word2vec_model.key_to_index}

glove_words = list(glove_model.keys())
glove_vectors = np.array(list(glove_model.values()))


def decode_vector_to_glove_word(vector, glove_words, glove_vectors):
    # Compute cosine similarity between the aligned vector and all GloVe vectors
    similarities = cosine_similarity(vector.reshape(1, -1), glove_vectors)
    # Find the index of the closest GloVe vector
    closest_index = np.argmax(similarities)
    return glove_words[closest_index]


counter = 0
for word in tqdm(common_vocab):
    aligned_vector = aligned_word2vec[word]  # This is an aligned vector in GloVe space

    decoded_word = decode_vector_to_glove_word(aligned_vector, glove_words, glove_vectors)
    if word != decoded_word:
        print(f"The decoded word for the aligned vector is: {decoded_word}, original {word}")
        counter += 1

print(f"mismatched: {counter / len(common_vocab)}")
