import numpy as np
from sklearn.datasets import fetch_20newsgroups
import torch
from torch.nn.functional import cosine_similarity

def retrieve_documents(query_embedding, doc_embeddings, doc_labels):
    similarities = cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
    return sorted(range(len(doc_embeddings)), key=lambda i: similarities[i], reverse=True)


def evaluate(retrieved_indices, ground_truth_indices):
    # Assuming a binary relevance (relevant or not relevant)
    retrieved = set(retrieved_indices)
    ground_truth = set(ground_truth_indices)
    precision = len(retrieved.intersection(ground_truth)) / len(retrieved)
    recall = len(retrieved.intersection(ground_truth)) / len(ground_truth)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score

# Load the dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# np.save('20news_glove_train.npy', Xtr)
# np.save('20news_glove_test.npy', Xte)
doc_embeddings, query_embeddings = np.load('20news_glove_train.npy'), np.load('20news_glove_test.npy')
doc_embeddings, query_embeddings = torch.from_numpy(doc_embeddings), torch.from_numpy(query_embeddings)
doc_labels, query_labels = newsgroups_train.target, newsgroups_test.target

# Evaluate multiple queries
all_precisions = []
all_recalls = []
all_f1_scores = []

for query_embedding, query_label in zip(query_embeddings, query_labels):
    retrieved_indices = retrieve_documents(query_embedding, doc_embeddings, doc_labels)
    ground_truth_indices = [i for i, label in enumerate(doc_labels) if label == query_label]

    precision, recall, f1_score = evaluate(retrieved_indices, ground_truth_indices)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1_score)

# Aggregate the metrics
avg_precision = sum(all_precisions) / len(all_precisions)
avg_recall = sum(all_recalls) / len(all_recalls)
avg_f1_score = sum(all_f1_scores) / len(all_f1_scores)

print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1_score}")
