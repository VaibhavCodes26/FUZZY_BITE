from fuzzy_logic.inference_engine import InferenceEngine

class RecommendationEngine:
    def __init__(self):
        self.inference_engine = InferenceEngine()

    def _calculate_price_suitability(self, restaurant_price, user_budget):
        """
        Converts actual price to a 0-100 suitability score based on budget.
        """
        if restaurant_price <= user_budget:
            return 100.0
        elif restaurant_price >= (user_budget * 2):
            return 0.0
        else:
            # Linear decay from 100 to 0 as price goes from budget to 2x budget
            penalty = ((restaurant_price - user_budget) / user_budget) * 100
            return 100.0 - penalty

    def _calculate_cuisine_match(self, restaurant_cuisine, user_cuisine):
        """
        Returns crisp cuisine match degree which will be fuzzified by the engine.
        Unspecified cuisine is treated as neutral (50).
        """
        if not user_cuisine or user_cuisine.lower() == "any":
            return 50.0
        if restaurant_cuisine.lower() == user_cuisine.lower():
            return 100.0
        return 0.0

    def recommend(self, restaurants, user_prefs):
        """
        Filters by hard constraints, evaluates fuzzy scores, and ranks results.
        """
        eligible_restaurants = []

        # 1. Hard Constraint Filtering
        for r in restaurants:
            # Dietary constraint
            if user_prefs.get('req_veg', False) and not r.get('is_veg', False):
                continue
            
            # Strict maximum distance limit
            if r['distance'] > user_prefs.get('max_distance', 15.0):
                continue
                
            eligible_restaurants.append(r)

        if not eligible_restaurants:
            return {"count": 0, "results": [], "message": "No restaurants meet your strict constraints."}

        # 2. Fuzzy Scoring
        scored_results = []
        for r in eligible_restaurants:
            p_suitability = self._calculate_price_suitability(r['price'], user_prefs.get('budget', 500))
            c_match = self._calculate_cuisine_match(r['cuisine'], user_prefs.get('cuisine', 'any'))
            
            inference_result = self.inference_engine.evaluate(
                rating=r['rating'],
                price_suitability=p_suitability,
                distance=r['distance'],
                cuisine_match=c_match
            )
            
            scored_results.append({
                "id": r['id'],
                "name": r['name'],
                "fuzzy_score": inference_result['score'],
                "inference_details": inference_result
            })

        # 3. Deterministic Ranking
        # Sort by fuzzy score descending. Tie-breaker is the restaurant ID to ensure reproducibility.
        scored_results.sort(key=lambda x: (x['fuzzy_score'], -x['id']), reverse=True)

        return {
            "count": len(scored_results),
            "results": scored_results,
            "message": "Recommendations generated successfully."
        }