import pytest
from fuzzy_logic.recommendation_engine import RecommendationEngine

@pytest.fixture
def recommender():
    return RecommendationEngine()

@pytest.fixture
def sample_restaurants():
    return [
        {"id": 1, "name": "Veg Paradise", "price": 400, "distance": 2.0, "rating": 4.5, "cuisine": "Indian", "is_veg": True},
        {"id": 2, "name": "Meat Grill", "price": 800, "distance": 1.5, "rating": 4.2, "cuisine": "American", "is_veg": False},
        {"id": 3, "name": "Far Cafe", "price": 300, "distance": 12.0, "rating": 4.0, "cuisine": "Italian", "is_veg": True},
        {"id": 4, "name": "Tie Breaker 1", "price": 500, "distance": 5.0, "rating": 3.0, "cuisine": "Any", "is_veg": True},
        {"id": 5, "name": "Tie Breaker 2", "price": 500, "distance": 5.0, "rating": 3.0, "cuisine": "Any", "is_veg": True}
    ]

def test_hard_constraints_filtering(recommender, sample_restaurants):
    user_prefs = {"budget": 500, "max_distance": 5.0, "cuisine": "Any", "req_veg": True}
    result = recommender.recommend(sample_restaurants, user_prefs)
    
    # ID 2 is non-veg, ID 3 is too far (12.0 > 5.0)
    assert result['count'] == 3
    valid_ids = [r['id'] for r in result['results']]
    assert 1 in valid_ids
    assert 2 not in valid_ids
    assert 3 not in valid_ids

def test_empty_eligible_set(recommender, sample_restaurants):
    # Budget too low, distance too tight
    user_prefs = {"budget": 100, "max_distance": 0.5, "cuisine": "Any", "req_veg": True}
    result = recommender.recommend(sample_restaurants, user_prefs)
    assert result['count'] == 0
    assert result['results'] == []

def test_tie_breaking(recommender, sample_restaurants):
    user_prefs = {"budget": 1000, "max_distance": 15.0, "cuisine": "Any", "req_veg": True}
    result = recommender.recommend(sample_restaurants, user_prefs)
    
    # Extract IDs 4 and 5 which have identical inputs
    tie_results = [r for r in result['results'] if r['id'] in [4, 5]]
    
    # Check deterministic tie breaker (Reverse ID order)
    assert tie_results[0]['id'] == 5
    assert tie_results[1]['id'] == 4