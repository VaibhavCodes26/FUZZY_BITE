import pytest
from evaluation.metrics import precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k

def test_precision_at_k():
    labels = [3, 2, 0, 1, 0] # Relevance labels of retrieved items
    # k=3, threshold=1. Top 3 are [3, 2, 0] -> 2 relevant
    assert precision_at_k(labels, k=3, threshold=1) == 2/3
    # k=0
    assert precision_at_k(labels, k=0) == 0.0

def test_recall_at_k():
    labels = [3, 2, 0, 1, 0]
    total_relevant = 5 
    assert recall_at_k(labels, total_relevant, k=3, threshold=1) == 2/5

def test_ndcg_at_k():
    perfect_ranking = [4, 3, 2, 1]
    worst_ranking = [1, 2, 3, 4]
    
    # NDCG of perfect ranking must be exactly 1.0
    assert pytest.approx(ndcg_at_k(perfect_ranking, k=4), 0.001) == 1.0
    
    # Worst ranking should be less than 1.0
    assert ndcg_at_k(worst_ranking, k=4) < 1.0
    
    # Empty case
    assert ndcg_at_k([], k=3) == 0.0