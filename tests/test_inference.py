import pytest
from fuzzy_logic.inference_engine import InferenceEngine

@pytest.fixture
def engine():
    return InferenceEngine()

def test_inference_bounds(engine):
    """Ensures defuzzified output never exceeds the 0-100 scale."""
    result = engine.evaluate(rating=5, price_suitability=100, distance=0, cuisine_match=100)
    assert 0 <= result['score'] <= 100

def test_perfect_scenario_score(engine):
    """An ideal restaurant should trigger 'Very High' rules and yield a high centroid score."""
    result = engine.evaluate(rating=4.8, price_suitability=95, distance=1, cuisine_match=95)
    # Centroid of a right-skewed output MF usually maxes around 85-95, not exactly 100
    assert result['score'] > 75 
    assert result['status'] == "Success"

def test_poor_scenario_score(engine):
    """A terrible restaurant should trigger 'Very Low' rules and yield a low score."""
    result = engine.evaluate(rating=1, price_suitability=10, distance=14, cuisine_match=10)
    assert result['score'] < 30

def test_determinism(engine):
    """Identical inputs must yield exactly identical defuzzified outputs."""
    res1 = engine.evaluate(rating=3.5, price_suitability=60, distance=5, cuisine_match=60)
    res2 = engine.evaluate(rating=3.5, price_suitability=60, distance=5, cuisine_match=60)
    assert res1['score'] == res2['score']

def test_out_of_bounds_input_clipping(engine):
    """Engine should handle out-of-bounds inputs by clipping without crashing."""
    res1 = engine.evaluate(rating=6.0, price_suitability=150, distance=-5, cuisine_match=120)
    res_clipped = engine.evaluate(rating=5.0, price_suitability=100, distance=0, cuisine_match=100)
    assert res1['score'] == res_clipped['score']