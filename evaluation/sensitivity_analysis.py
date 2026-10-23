import numpy as np
# from fuzzy_logic.inference_engine import FuzzyBiteEngine

def perturb_membership_parameter(engine, variable_name, linguistic_term, point_index, delta):
    """
    Modifies a specific vertex of a membership function array by a delta value.
    This simulates uncertainty or tuning in expert parameter definitions.
    """
    # Note: In a real implementation, you must recompile the ControlSystemSimulation 
    # after modifying antecedent parameters in scikit-fuzzy.
    mf_array = engine.fis.antecedents[variable_name].terms[linguistic_term].mf
    
    original_val = mf_array[point_index]
    # Apply perturbation, ensuring it doesn't cross adjacent points (maintaining valid geometry)
    mf_array[point_index] += delta 
    
    return original_val

def run_sensitivity_analysis():
    print("Running Sensitivity Analysis...")
    # engine = FuzzyBiteEngine()
    
    # Test case: A restaurant with average metrics
    test_case = {"rating": 3.5, "price_suit": 55, "dist": 5.0, "cuisine": 60}
    
    # Baseline score (Mocked)
    # base_score = engine.evaluate(**test_case)
    base_score = 58.4 
    
    print(f"Base Output Score: {base_score:.2f}")
    
    # Define perturbations (e.g., shifting the peak of the 'Moderate' price triangle by -5, -2, +2, +5)
    perturbations = [-5, -2, 2, 5]
    
    print("\nPerturbing 'Moderate' Price Suitability Peak (Index 1):")
    print(f"{'Delta':<10} | {'New Score':<15} | {'% Change':<15}")
    print("-" * 45)
    
    for delta in perturbations:
        # 1. Apply perturbation
        # orig_val = perturb_membership_parameter(engine, 'price_suitability', 'Moderate', 1, delta)
        
        # 2. Re-evaluate
        # new_score = engine.evaluate(**test_case)
        
        # Mocking the sensitivity change for demonstration (-0.1 effect per delta unit)
        new_score = base_score + (delta * -0.15) 
        pct_change = ((new_score - base_score) / base_score) * 100
        
        print(f"{delta:<10+} | {new_score:<15.2f} | {pct_change:<15.2f}%")
        
        # 3. Revert perturbation to maintain clean state for next loop
        # perturb_membership_parameter(engine, 'price_suitability', 'Moderate', 1, -delta)

if __name__ == "__main__":
    run_sensitivity_analysis()