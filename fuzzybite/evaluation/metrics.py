import numpy as np

def precision_at_k(relevance_labels, k, threshold=1):
    if k == 0 or len(relevance_labels) == 0: return 0.0
    top_k = relevance_labels[:k]
    return sum(1 for label in top_k if label >= threshold) / len(top_k)

def recall_at_k(relevance_labels, all_relevant_count, k, threshold=1):
    if k == 0 or all_relevant_count == 0: return 0.0
    top_k = relevance_labels[:k]
    return sum(1 for label in top_k if label >= threshold) / all_relevant_count

def dcg_at_k(relevance_labels, k):
    # Updated for NumPy 2.0 compatibility
    r = np.asarray(relevance_labels, dtype=float)[:k]
    if r.size == 0: return 0.0
    return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(relevance_labels, k):
    dcg = dcg_at_k(relevance_labels, k)
    idcg = dcg_at_k(sorted(relevance_labels, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0
