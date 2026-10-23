import csv
from flask import Flask, render_template, request, jsonify
from fuzzy_logic.recommendation_engine import RecommendationEngine

app = Flask(__name__)
recommender = RecommendationEngine()

def load_restaurants():
    """Loads the synthetic dataset from the CSV file."""
    restaurants = []
    with open('data/restaurants.csv', mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            restaurants.append({
                "id": int(row["id"]),
                "name": row["name"],
                "cuisine": row["cuisine"],
                "price": float(row["price"]),
                "rating": float(row["rating"]),
                "distance": float(row["distance"]),
                "is_veg": row["is_veg"].strip().lower() == 'true'
            })
    return restaurants

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    user_prefs = request.json
    restaurants = load_restaurants()
    
    # Run the fuzzy logic recommendation engine
    results = recommender.recommend(restaurants, user_prefs)
    return jsonify(results)

if __name__ == '__main__':
    print("Starting FuzzyBite Server...")
    app.run(debug=True, port=5000)