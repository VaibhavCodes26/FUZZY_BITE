import numpy as np
from skfuzzy import control as ctrl
from fuzzy_logic.membership_functions import define_variables
from fuzzy_logic.rule_base import build_rule_base

class InferenceEngine:
    def __init__(self):
        self.variables = define_variables()
        self.rules = build_rule_base(self.variables)
        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def evaluate(self, rating, price_suitability, distance, cuisine_match):
        """
        Executes the Mamdani fuzzy inference process for a single restaurant.
        Returns the defuzzified score and inference details.
        """
        # 1. Clamp inputs to universes
        self.simulator.input['rating'] = np.clip(rating, 0, 5)
        self.simulator.input['price_suitability'] = np.clip(price_suitability, 0, 100)
        self.simulator.input['distance'] = np.clip(distance, 0, 15)
        self.simulator.input['cuisine_match'] = np.clip(cuisine_match, 0, 100)

        try:
            # 2. Compute inference (Fuzzification -> Aggregation -> Defuzzification)
            self.simulator.compute()
            score = self.simulator.output['suitability']
            
            # Scikit-Fuzzy does not easily expose per-rule firing strengths in the public API 
            # after computation without internal hacking. We document successful activation.
            status = "Success"
        except ValueError:
            # Fallback policy: If no rules activate (sparse rule base coverage for the specific input),
            # default to a heavily penalized neutral score.
            score = 0.0
            status = "No rules activated. Fallback applied."

        return {
            "score": round(float(score), 2),
            "status": status,
            "inputs": {
                "rating": rating,
                "price_suitability": price_suitability,
                "distance": distance,
                "cuisine_match": cuisine_match
            }
        }