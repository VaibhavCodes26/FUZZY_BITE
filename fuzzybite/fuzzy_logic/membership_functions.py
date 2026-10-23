import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def define_variables():
    rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'rating')
    price_suitability = ctrl.Antecedent(np.arange(0, 101, 1), 'price_suitability')
    distance = ctrl.Antecedent(np.arange(0, 15.1, 0.1), 'distance')
    cuisine_match = ctrl.Antecedent(np.arange(0, 101, 1), 'cuisine_match')
    suitability = ctrl.Consequent(np.arange(0, 101, 1), 'suitability')
    suitability.defuzzify_method = 'centroid'

    rating['Low'] = fuzz.trapmf(rating.universe, [0, 0, 2, 3])
    rating['Average'] = fuzz.trimf(rating.universe, [2, 3.5, 4.5])
    rating['High'] = fuzz.trapmf(rating.universe, [3.5, 4.5, 5, 5])

    price_suitability['Poor'] = fuzz.trapmf(price_suitability.universe, [0, 0, 30, 50])
    price_suitability['Moderate'] = fuzz.trimf(price_suitability.universe, [30, 50, 70])
    price_suitability['Good'] = fuzz.trapmf(price_suitability.universe, [50, 80, 100, 100])

    distance['Near'] = fuzz.trapmf(distance.universe, [0, 0, 2, 5])
    distance['Moderate'] = fuzz.trimf(distance.universe, [3, 6, 10])
    distance['Far'] = fuzz.trapmf(distance.universe, [8, 12, 15, 15])

    cuisine_match['Poor'] = fuzz.trimf(cuisine_match.universe, [0, 0, 40])
    cuisine_match['Partial'] = fuzz.trimf(cuisine_match.universe, [30, 50, 70])
    cuisine_match['Excellent'] = fuzz.trimf(cuisine_match.universe, [60, 100, 100])

    suitability['Very Low'] = fuzz.trapmf(suitability.universe, [0, 0, 15, 25])
    suitability['Low'] = fuzz.trimf(suitability.universe, [15, 30, 45])
    suitability['Moderate'] = fuzz.trimf(suitability.universe, [35, 50, 65])
    suitability['High'] = fuzz.trimf(suitability.universe, [55, 70, 85])
    suitability['Very High'] = fuzz.trapmf(suitability.universe, [75, 85, 100, 100])

    return {'rating': rating, 'price_suitability': price_suitability, 'distance': distance, 'cuisine_match': cuisine_match, 'suitability': suitability}
