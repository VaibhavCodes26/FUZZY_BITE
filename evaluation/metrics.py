import numpy as np

def precision_at_k(relevance_labels, k, threshold=1):
    """
    Calculates Precision at k.
    A recommendation is considered relevant if its label >= threshold.
    """
    if k == 0 or len(relevance_labels) == 0:
        return 0.0
    
    top_k = relevance_labels[:k]
    relevant_count = sum(1 for label in top_k if label >= threshold)
    return relevant_count / len(top_k)

def recall_at_k(relevance_labels, all_relevant_count, k, threshold=1):
    """
    Calculates Recall at k.
    all_relevant_count is the total number of relevant items in the entire dataset for this query.
    """
    if k == 0 or all_relevant_count == 0:
        return 0.0
    
    top_k = relevance_labels[:k]
    relevant_count = sum(1 for label in top_k if label >= threshold)
    return relevant_count / all_relevant_count

def dcg_at_k(relevance_labels, k):
    """
    Calculates Discounted Cumulative Gain at k.
    Uses the standard formula: sum( (2^rel - 1) / log2(rank + 1) )
    """
    r = np.asfarray(relevance_labels)[:k]
    if r.size == 0:
        return 0.0
    return np.sum((np.power(2, r) - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(relevance_labels, k):
    """
    Calculates Normalized Discounted Cumulative Gain at k.
    """
    dcg = dcg_at_k(relevance_labels, k)
    ideal_relevance_labels = sorted(relevance_labels, reverse=True)
    idcg = dcg_at_k(ideal_relevance_labels, k)
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg