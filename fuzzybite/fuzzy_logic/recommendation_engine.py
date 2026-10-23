from fuzzy_logic.inference_engine import InferenceEngine

class RecommendationEngine:
    def __init__(self):
        self.inference_engine = InferenceEngine()

    def _calculate_price_suitability(self, restaurant_price, user_budget):
        if restaurant_price <= user_budget:
            return 100.0
        elif restaurant_price >= (user_budget * 2):
            return 0.0
        else:
            penalty = ((restaurant_price - user_budget) / user_budget) * 100
            return 100.0 - penalty

    def _calculate_cuisine_match(self, restaurant_cuisine, user_cuisine):
        if not user_cuisine or user_cuisine.lower() == "any":
            return 50.0
        if restaurant_cuisine.lower() == user_cuisine.lower():
            return 100.0
        return 0.0

    def recommend(self, restaurants, user_prefs):
        eligible_restaurants = []
        for r in restaurants:
            if user_prefs.get('req_veg', False) and not r.get('is_veg', False):
                continue
            if r['distance'] > user_prefs.get('max_distance', 15.0):
                continue
            eligible_restaurants.append(r)

        if not eligible_restaurants:
            return {"count": 0, "results": [], "message": "No restaurants meet constraints."}

        scored_results = []
        for r in eligible_restaurants:
            p_suit = self._calculate_price_suitability(r['price'], user_prefs.get('budget', 500))
            c_match = self._calculate_cuisine_match(r['cuisine'], user_prefs.get('cuisine', 'any'))
            
            inference_result = self.inference_engine.evaluate(
                rating=r['rating'],
                price_suitability=p_suit,
                distance=r['distance'],
                cuisine_match=c_match
            )
            
            scored_results.append({
                "id": r['id'],
                "name": r['name'],
                "fuzzy_score": inference_result['score'],
                "inference_details": inference_result
            })

        scored_results.sort(key=lambda x: (x['fuzzy_score'], -x['id']), reverse=True)
        return {"count": len(scored_results), "results": scored_results, "message": "Success."}
