import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

from tqdm import tqdm

# Download NLTK stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    return tokens

def document_to_glove_vector(document, glove_embeddings, embedding_dim=100):
    tokens = preprocess(document)
    document_vec = np.zeros(embedding_dim, dtype='float32')
    num_tokens = 0
    for token in tokens:
        if token in glove_embeddings:
            document_vec += glove_embeddings[token]
            num_tokens += 1
    if num_tokens > 0:
        document_vec /= num_tokens

    return document_vec

# Load the dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

glove_embeddings = np.load('vanilla_glove_100D.npy', allow_pickle=True).item()
# glove_embeddings = np.load('poincare_glove_100D_cosh-dist-sq_init_trick.npy', allow_pickle=True).item()

# Example: Convert the first document to GloVe vector
train_docs, test_docs = newsgroups_train.data, newsgroups_test.data

Xtr = np.zeros((len(train_docs), 100), dtype='float32')
Xte = np.zeros((len(test_docs), 100), dtype='float32')
for i, doc in tqdm(enumerate(train_docs)):
    Xtr[i] = document_to_glove_vector(doc, glove_embeddings)
for i, doc in tqdm(enumerate(test_docs)):
    Xte[i] = document_to_glove_vector(doc, glove_embeddings)

np.save('20news_glove_train.npy', Xtr)
np.save('20news_glove_test.npy', Xte)

# np.save('20news_glovehyp_train.npy', Xtr)
# np.save('20news_glovehyp_test.npy', Xte)
