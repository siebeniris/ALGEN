import itertools
import os.path
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

import gensim.downloader as api

np.random.seed(42)


def load_word_vectors(model_name):
    if model_name == "glove100":
        return api.load("glove-wiki-gigaword-100")
    elif model_name == "glove300":
        return api.load("glove-wiki-gigaword-300")
    elif model_name == "word2vec":
        return KeyedVectors.load_word2vec_format("vectors/GoogleNews-vectors-negative300.bin", binary=True)

    else:
        print("Please choose [glove100], [glove300], [word2vec]")


def get_train_test_data(glove100, glove300, word2vec, test_samples=300):
    common_vocab = set(glove100.key_to_index.keys()).intersection(glove300.key_to_index.keys())
    common_vocab = set(word2vec.key_to_index.keys()).intersection(common_vocab)
    print(f"There are {len(common_vocab)} common vocab.")
    common_vocab = list(common_vocab)

    # Define batch size
    word2vec_matrix = np.array([word2vec[word] for word in common_vocab])
    glove100_matrix = np.array([glove100[word] for word in common_vocab])
    glove300_matrix = np.array([glove300[word] for word in common_vocab])

    test_vocab = common_vocab[-test_samples:]
    test_word2vec = word2vec_matrix[-test_samples:]
    test_glove100 = glove100_matrix[-test_samples:]
    test_glove300 = glove300_matrix[-test_samples:]

    word2vec_matrix = word2vec_matrix[:-test_samples]
    glove100_matrix = glove100_matrix[:-test_samples]
    glove300_matrix = glove300_matrix[:-test_samples]

    return (test_glove100, test_glove300, test_word2vec,
            glove100_matrix, glove300_matrix, word2vec_matrix, test_vocab)


def test_alignment(word, target_words, target_vectors, aligned_dict):
    """
    Test the alignment of embeddings for a given word.

    Args:
        word (str): The word to test.
        glove_embeddings (dict): GloVe embeddings.
        aligned_dict (dict): Aligned embeddings.

    Returns:
        tuple: Closest word in aligned space and its similarity to the target word.
    """
    # Ensure the word exists in both embeddings
    if word not in target_words:
        print(f"Word '{word}' is missing in target embeddings.")
        return None, None

    aligned_embedding = aligned_dict[word]

    # it is not really mapped to the target embeddings space.
    similarities = cosine_similarity([aligned_embedding], target_vectors)[0]

    closest_index = np.argmax(similarities)
    close_sim = np.max(similarities)
    closet_word = target_words[closest_index]
    return closet_word, close_sim


def get_cosine_similarities_3d(X, Y):
    cosine_similarities = np.array([
        cosine_similarity(X[i].reshape(-1, 1).T, Y[i].reshape(-1, 1).T)[0, 0]
        for i in range(X.shape[0])])
    return np.mean(cosine_similarities)


def mapping_source_to_target_space(X, Y, test_X, test_Y,
                                   test_vocab,
                                   sample_size,
                                   slice_x=None, shift="left"):
    ## this is one-off inverse. no need batches
    aligned_batches = []
    test_batches = []

    if slice_x is not None:
        if shift == "left":
            x = X[:sample_size][:, :slice_x]
        elif shift == "right":
            x = X[:sample_size][:, slice_x:]
        else:
            x = X[:sample_size][:, :slice_x]

    else:
        x = X[:sample_size]
    y = Y[:sample_size]

    test_x = test_X
    test_y = test_Y
    A = np.linalg.pinv(x.T @ x) @ x.T @ y  # (d_s, d_t)
    # print(A.shape)
    # X = A+ Y
    if slice_x is not None:
        if shift == "left":
            aligned_test_x = test_x[:, :slice_x] @ A
        elif shift == "right":
            aligned_test_x = test_x[:, slice_x:] @ A
        else:
            aligned_test_x = test_x[:, :slice_x] @ A
    else:
        aligned_test_x = test_x @ A

    X_aligned = x @ A
    # print(aligned_test_x.shape)
    # print("if aligned x equals to y:", np.sum(X_aligned == y))  # it is not the same.
    aligned_batches.append(X_aligned)

    # print("if aligned test x equals to y:", np.sum(aligned_test_x == test_y))
    test_batches.append(aligned_test_x)
    # Combine aligned batches
    X_aligned = np.vstack(aligned_batches)

    x_test_aligned = np.vstack(test_batches)

    # get cosine similarity of aligned X and Y
    X_Y_cossim = get_cosine_similarities_3d(X_aligned, Y)

    test_vectors = test_y

    assert len(test_vocab) == len(test_vectors)
    test_aligned_dict = {word: x_test_aligned[i] for i, word in enumerate(test_vocab)}

    ################################################################
    # test alignment.
    correct_words = []
    cosine_sims = []
    counter = 0
    for word in test_aligned_dict:
        # find the nearest.
        res = test_alignment(word, test_vocab, test_vectors, test_aligned_dict)
        if res is not None:
            closest_word, score = res
            cosine_sims.append(score)

            if closest_word == word:
                # print("correct!!!!:", word, "closest word in Glove: ", closest_word, " score:", score)
                correct_words.append(word)
            # else:
            # print("wrong:", word, "closest word in Glove: ", closest_word, " score:", score)
            counter += 1
    # print("accuracy:", len(correct_words) / counter, "cosine mean: ", np.mean(cosine_sims))
    return len(correct_words) / counter, np.mean(cosine_sims), X_Y_cossim


def iterate_batch_sizes(X, Y, test_X, test_Y, test_vocab, writer):
    total_samples = X.shape[0]
    best_acc = -float("inf")
    patience = 5000
    no_improvement_since = 0
    for batch_size in range(100, 1000, 100):
        test_accuracy, test_cosine_sim, X_Y_cossim = mapping_source_to_target_space(X, Y, test_X, test_Y,
                                                                                    test_vocab,
                                                                                    batch_size,
                                                                                    slice_x=None)
        print(
            f"X size {batch_size}, test accuracy: {test_accuracy},test cossim: {test_cosine_sim}, train cossim: {X_Y_cossim}")
        writer.write(f"{batch_size},{test_accuracy},{test_cosine_sim},{X_Y_cossim}\n")

    for batch_size in range(1000, total_samples, 1000):
        test_accuracy, test_cosine_sim, X_Y_cossim = mapping_source_to_target_space(X, Y, test_X, test_Y,
                                                                                    test_vocab,
                                                                                    batch_size,
                                                                                    slice_x=None)
        print(
            f"X size {batch_size}, test accuracy: {test_accuracy},test cossim: {test_cosine_sim}, train cossim: {X_Y_cossim}")
        writer.write(f"{batch_size},{test_accuracy},{test_cosine_sim},{X_Y_cossim}\n")

        if batch_size > 10000:
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                no_improvement_since = 0
            else:
                no_improvement_since += 1000

            if no_improvement_since >= patience:
                print("Early stopping triggered due to no improvement in test accuracy")
                break


def find_optimal_dimension(X, Y, test_X, test_Y, test_vocab, batch_size, writer):
    dim = X.shape[-1]
    for slice_x in range(5, dim, 5):
        for shift in ["left", "right"]:
            test_accuracy, test_cosine_sim, X_Y_cossim = mapping_source_to_target_space(X, Y, test_X, test_Y,
                                                                                        test_vocab,
                                                                                        batch_size,
                                                                                        slice_x=slice_x, shift=shift)
            print(
                f"X size {batch_size}, shift {shift}, dim {slice_x}, test accuracy: {test_accuracy}, "
                f"test cossim: {test_cosine_sim}, "
                f"train cossim: {X_Y_cossim}")
            writer.write(f"{batch_size},{shift},{slice_x},{test_accuracy},{test_cosine_sim},{X_Y_cossim}\n")


def main(output_dir="results/normal_equations/wordvectors"):
    print('Loading Word2vec ...')
    word2vec = load_word_vectors("word2vec")
    print("loading Glove100 ...")
    glove100 = load_word_vectors("glove100")
    print("Loading Glove300 ...")
    glove300 = load_word_vectors("glove300")

    (test_glove100, test_glove300, test_word2vec,
     glove100_matrix, glove300_matrix, word2vec_matrix, test_vocab) = \
        get_train_test_data(
            glove100, glove300, word2vec, test_samples=300)

    name2model = {"word2vec": [word2vec_matrix, test_word2vec],
                  "glove100": [glove100_matrix, test_glove100],
                  "glove300": [glove300_matrix, test_glove300]}

    for p in itertools.permutations(["word2vec", "glove100", "glove300"], 2):
        a, b = p
        print(f"Mapping source vectors {a} to target vectors {b}...")
        X, test_X = name2model[a]
        Y, test_Y = name2model[b]
        print(f"from {X.shape[-1]} to {Y.shape[-1]} dimension")

        # f_writer = open(os.path.join(output_dir, f"{a}_{b}.csv"), "a+")
        # f_writer.write("samples,test_acc,test_cossim,X_Y_cossim\n")
        # iterate_batch_sizes(X, Y, test_X, test_Y, test_vocab, f_writer)
        # f_writer.close()

        f_writer = open(os.path.join(output_dir, f"{a}_{b}_samples_6000.csv"), "a+")
        f_writer.write("samples,shift,dim,test_acc,test_cossim,X_Y_cossim\n")
        find_optimal_dimension(X, Y, test_X, test_Y, test_vocab, batch_size=6000, writer=f_writer)
        f_writer.close()


if __name__ == '__main__':
    main()
