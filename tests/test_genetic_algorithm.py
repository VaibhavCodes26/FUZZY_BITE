import pytest
import numpy as np
from optimization.genetic_algorithm import FuzzyBiteGA

@pytest.fixture
def mock_ga():
    # Use a dummy path since we just test unit math here
    return FuzzyBiteGA(scenarios_path="dummy.json")

def test_chromosome_repair(mock_ga):
    """
    Ensures that triangular membership functions maintain the invariant a <= b <= c.
    """
    # Bad chromosome: left vertex is higher than peak, right vertex is lower than peak
    # Low: [30, 20, 10] -> Should repair to [20, 20, 20] or similar valid structure
    invalid_chrom = np.array([30, 20, 10,  70, 50, 30,  150, 70, -10], dtype=float)
    
    repaired = mock_ga._repair_chromosome(invalid_chrom)
    
    # Test clipping bounds (0-100)
    assert np.max(repaired) <= 100
    assert np.min(repaired) >= 0
    
    # Test valid geometry a <= b <= c for Low (indices 0,1,2)
    assert repaired[0] <= repaired[1]
    assert repaired[1] <= repaired[2]
    
    # Test valid geometry a <= b <= c for Moderate (indices 3,4,5)
    assert repaired[3] <= repaired[4]
    assert repaired[4] <= repaired[5]
    
    # Test valid geometry a <= b <= c for High (indices 6,7,8)
    assert repaired[6] <= repaired[7]
    assert repaired[7] <= repaired[8]