import pytest
import numpy as np
import skfuzzy as fuzz
from fuzzy_logic.membership_functions import define_variables

@pytest.fixture
def variables():
    return define_variables()

def test_universe_bounds(variables):
    """Ensures universes of discourse are defined within correct hard limits."""
    assert np.min(variables['rating'].universe) == 0
    assert np.max(variables['rating'].universe) == 5.0
    assert np.max(variables['price_suitability'].universe) == 100
    assert np.max(variables['distance'].universe) == 15.0
    assert np.max(variables['cuisine_match'].universe) == 100
    assert np.max(variables['suitability'].universe) == 100

def test_membership_degrees(variables):
    """Tests if the membership degree is calculated correctly at peak and boundary points."""
    P = variables['price_suitability']
    
    # Extract the 'Moderate' triangular membership array
    mod_mf = P.terms['Moderate'].mf
    
    # Calculate degree of membership at specific crisp values
    degree_at_30 = fuzz.interp_membership(P.universe, mod_mf, 30)
    degree_at_50 = fuzz.interp_membership(P.universe, mod_mf, 50)
    degree_at_70 = fuzz.interp_membership(P.universe, mod_mf, 70)
    
    # At boundaries it should be 0, at peak it should be 1
    assert degree_at_30 == 0.0
    assert degree_at_50 == 1.0
    assert degree_at_70 == 0.0

def test_overlapping_membership(variables):
    """Ensures that cross-over points sum to approximately 1 (or >0) representing valid overlap."""
    D = variables['distance']
    near_mf = D.terms['Near'].mf
    mod_mf = D.terms['Moderate'].mf
    
    # At distance 4.0, it should belong partially to Near and Moderate
    degree_near = fuzz.interp_membership(D.universe, near_mf, 4.0)
    degree_mod = fuzz.interp_membership(D.universe, mod_mf, 4.0)
    
    assert degree_near > 0
    assert degree_mod > 0