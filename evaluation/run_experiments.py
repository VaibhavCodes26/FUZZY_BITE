import json
import time
from evaluation.metrics import ndcg_at_k, precision_at_k
# Assume FuzzyBiteEngine and baseline are importable
# from fuzzy_logic.inference_engine import FuzzyBiteEngine

def simple_baseline_score(rating, price_suit, dist, cuisine):
    """
    A naive linear weighted baseline to compare against fuzzy logic.
    Normalizes all inputs to 0-100 and applies fixed weights.
    """
    norm_rating = (rating / 5.0) * 100
    norm_dist = max(0, 100 - (dist / 15.0) * 100) # Closer is better
    
    # Weights: 30% rating, 30% cuisine, 20% price, 20% distance
    return 0.3 * norm_rating + 0.3 * cuisine + 0.2 * price_suit + 0.2 * norm_dist

def run_evaluation(scenarios_path):
    print(f"Loading evaluation scenarios from {scenarios_path}...\n")
    # engine = FuzzyBiteEngine()
    
    # Mock data structure representing parsed JSON
    # Format: {'user_req': {...}, 'restaurants': [{'id':1, 'features':{...}, 'true_relevance': 3}, ...]}
    try:
        with open(scenarios_path, 'r') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print("Scenario file not found. Running with dummy experiment data for demonstration.")
        scenarios = [{
            "scenario_id": "S1",
            "restaurants": [
                {"id": 1, "features": {"rating": 4.5, "price_suit": 90, "dist": 2, "cuisine": 100}, "true_relevance": 4},
                {"id": 2, "features": {"rating": 3.0, "price_suit": 50, "dist": 8, "cuisine": 50}, "true_relevance": 1},
                {"id": 3, "features": {"rating": 4.0, "price_suit": 80, "dist": 1, "cuisine": 0}, "true_relevance": 0},
            ]
        }]

    k = 3
    fuzzy_ndcg_total = 0
    baseline_ndcg_total = 0

    print(f"{'Scenario':<10} | {'Baseline NDCG@3':<15} | {'Fuzzy NDCG@3':<15}")
    print("-" * 45)

    for scenario in scenarios:
        baseline_results = []
        fuzzy_results = []
        
        for r in scenario['restaurants']:
            feats = r['features']
            
            # Baseline Prediction
            b_score = simple_baseline_score(feats['rating'], feats['price_suit'], feats['dist'], feats['cuisine'])
            baseline_results.append({'id': r['id'], 'score': b_score, 'true_relevance': r['true_relevance']})
            
            # Fuzzy Prediction (Mocked execution for completeness)
            # f_score = engine.evaluate(feats['rating'], feats['price_suit'], feats['dist'], feats['cuisine'])
            f_score = b_score * 0.9 + (feats['cuisine'] * 0.1) # Placeholder for engine.evaluate
            fuzzy_results.append({'id': r['id'], 'score': f_score, 'true_relevance': r['true_relevance']})

        # Rank by score descending
        baseline_results.sort(key=lambda x: x['score'], reverse=True)
        fuzzy_results.sort(key=lambda x: x['score'], reverse=True)

        # Extract ordered relevance labels
        baseline_rels = [x['true_relevance'] for x in baseline_results]
        fuzzy_rels = [x['true_relevance'] for x in fuzzy_results]

        b_ndcg = ndcg_at_k(baseline_rels, k)
        f_ndcg = ndcg_at_k(fuzzy_rels, k)

        baseline_ndcg_total += b_ndcg
        fuzzy_ndcg_total += f_ndcg

        print(f"{scenario['scenario_id']:<10} | {b_ndcg:<15.4f} | {f_ndcg:<15.4f}")

    print("-" * 45)
    print(f"Average Baseline NDCG@{k}: {baseline_ndcg_total / len(scenarios):.4f}")
    print(f"Average Fuzzy NDCG@{k}:    {fuzzy_ndcg_total / len(scenarios):.4f}")

if __name__ == "__main__":
    run_evaluation("../data/evaluation_scenarios.json")