import numpy as np


def load_glove_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        word_to_vec = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_to_vec[word] = vector
    return word_to_vec

glove_embeddings = load_glove_embeddings('poincare glove models/poincare_glove_100D_cosh-dist-sq_init_trick.txt')
print("Number of words in Glove embeddings = ", len(glove_embeddings))

# save_path = 'vanilla_glove_100D.npy'
save_path = 'poincare_glove_100D_cosh-dist-sq_init_trick.npy'
np.save(save_path, glove_embeddings)
