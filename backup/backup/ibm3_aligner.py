import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def ibm_model_3_with_sentence_cosine(source_sentences, target_sentences,
                                    source_embeddings, target_embeddings,
                                    iterations):
    def initialize_t_per_pair(source_sentence, target_sentence, source_embeddings, target_embeddings):
        t_e_f = {}
        source_matrix = np.array([source_embeddings[word] for word in source_sentence])
        target_matrix = np.array([target_embeddings[word] for word in target_sentence])
        cosine_matrix = cosine_similarity(source_matrix, target_matrix)

        for j, e_j in enumerate(source_sentence):
            t_e_f[e_j] = {}
            normalized_similarities = cosine_matrix[j] / np.sum(cosine_matrix[j])
            for i, f_i in enumerate(target_sentence):
                t_e_f[e_j][f_i] = normalized_similarities[i]
        return t_e_f

    # Initialize variables
    losses = []
    t_e_f_pairs = []
    for e, f in zip(source_sentences, target_sentences):
        t_e_f_pairs.append(initialize_t_per_pair(e, f, source_embeddings, target_embeddings))

    d_j_i_le_lf = lambda j, i, le, lf: 1 / lf
    p_0 = 0.2  # Null alignment probability

    # EM iterations
    for it in range(iterations):
        total_loss = 0
        for idx, (e, f) in enumerate(zip(source_sentences, target_sentences)):
            le, lf = len(e), len(f)
            t_e_f = t_e_f_pairs[idx]

            # E-Step
            c_e_f = {e_j: {f_i: 0 for f_i in f} for e_j in e}
            for j, e_j in enumerate(e):
                normalization = sum(t_e_f[e_j][f_i] * d_j_i_le_lf(j, i, le, lf) for i, f_i in enumerate(f))
                for i, f_i in enumerate(f):
                    p_aj_i = (t_e_f[e_j][f_i] * d_j_i_le_lf(j, i, le, lf)) / normalization
                    c_e_f[e_j][f_i] += p_aj_i
                total_loss -= np.log(normalization)

            # M-Step
            for e_j in e:
                for f_i in f:
                    t_e_f[e_j][f_i] = c_e_f[e_j][f_i] / sum(c_e_f[e_j].values())
            t_e_f_pairs[idx] = t_e_f

        # Record loss
        losses.append(total_loss)

    return t_e_f_pairs, losses
