import json
from evaluation.metrics import ndcg_at_k
from fuzzy_logic.inference_engine import InferenceEngine

def simple_baseline_score(rating, price_suit, dist, cuisine):
    # A naive linear weighted baseline to compare against fuzzy logic
    norm_rating = (rating / 5.0) * 100
    norm_dist = max(0, 100 - (dist / 15.0) * 100) 
    return 0.3 * norm_rating + 0.3 * cuisine + 0.2 * price_suit + 0.2 * norm_dist

def run_evaluation():
    engine = InferenceEngine()
    with open('data/evaluation_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    k = 3
    baseline_ndcg = 0
    fuzzy_ndcg = 0

    print(f"\n{'Scenario':<10} | {'Baseline NDCG@3':<15} | {'Fuzzy NDCG@3':<15}")
    print("-" * 45)

    for scenario in scenarios:
        b_res, f_res = [], []
        for r in scenario['restaurants']:
            feats = r['features']
            
            # Baseline Prediction
            b_score = simple_baseline_score(feats['rating'], feats['price_suit'], feats['dist'], feats['cuisine'])
            b_res.append({'score': b_score, 'rel': r['true_relevance']})
            
            # Fuzzy Logic Prediction
            f_score = engine.evaluate(feats['rating'], feats['price_suit'], feats['dist'], feats['cuisine'])['score']
            f_res.append({'score': f_score, 'rel': r['true_relevance']})

        # Rank by score descending
        b_res.sort(key=lambda x: x['score'], reverse=True)
        f_res.sort(key=lambda x: x['score'], reverse=True)

        # Calculate NDCG
        b_ndcg_val = ndcg_at_k([x['rel'] for x in b_res], k)
        f_ndcg_val = ndcg_at_k([x['rel'] for x in f_res], k)

        baseline_ndcg += b_ndcg_val
        fuzzy_ndcg += f_ndcg_val
        print(f"{scenario['scenario_id']:<10} | {b_ndcg_val:<15.4f} | {f_ndcg_val:<15.4f}")

    print("-" * 45)
    print(f"Average Baseline NDCG@{k}: {baseline_ndcg / len(scenarios):.4f}")
    print(f"Average Fuzzy NDCG@{k}:    {fuzzy_ndcg / len(scenarios):.4f}\n")

if __name__ == '__main__':
    run_evaluation()
