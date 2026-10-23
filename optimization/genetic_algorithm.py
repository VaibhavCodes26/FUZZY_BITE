import numpy as np
import json
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Assuming these are available in your project structure
from fuzzy_logic.membership_functions import define_variables
from fuzzy_logic.rule_base import build_rule_base
from evaluation.metrics import ndcg_at_k

class FuzzyBiteGA:
    def __init__(self, scenarios_path, pop_size=20, generations=15, mutation_rate=0.15, seed=42):
        self.scenarios_path = scenarios_path
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        np.random.seed(self.seed)
        
        # Load scenarios
        try:
            with open(self.scenarios_path, 'r') as f:
                self.scenarios = json.load(f)
        except FileNotFoundError:
            # Fallback for demonstration if file is missing
            self.scenarios = [{
                "scenario_id": "S1",
                "restaurants": [
                    {"id": 1, "features": {"rating": 4.5, "price_suit": 90, "dist": 2, "cuisine": 100}, "true_relevance": 4},
                    {"id": 2, "features": {"rating": 3.0, "price_suit": 50, "dist": 8, "cuisine": 50}, "true_relevance": 1},
                    {"id": 3, "features": {"rating": 4.0, "price_suit": 80, "dist": 1, "cuisine": 0}, "true_relevance": 0},
                ]
            }]

        # Base Chromosome: [Low_a, Low_b, Low_c, Mod_a, Mod_b, Mod_c, High_a, High_b, High_c]
        # These are the vertices for the Low, Moderate, and High triangular membership functions.
        self.base_chromosome = np.array([15, 30, 45, 35, 50, 65, 55, 70, 85], dtype=float)
        
        self.convergence_history = []

    def _repair_chromosome(self, chrom):
        """
        Constraint Handling: Ensures triangular validity (a <= b <= c) 
        and bounds within the 0-100 universe.
        """
        chrom = np.clip(chrom, 0, 100)
        
        # Low
        chrom[0] = min(chrom[0], chrom[1])
        chrom[2] = max(chrom[2], chrom[1])
        # Moderate
        chrom[3] = min(chrom[3], chrom[4])
        chrom[5] = max(chrom[5], chrom[4])
        # High
        chrom[6] = min(chrom[6], chrom[7])
        chrom[8] = max(chrom[8], chrom[7])
        
        return chrom

    def _init_population(self):
        """Generates the initial population by perturbing the base chromosome."""
        population = [self.base_chromosome] # Include baseline
        for _ in range(self.pop_size - 1):
            # Random perturbations between -10 and +10
            noise = np.random.uniform(-10, 10, size=len(self.base_chromosome))
            chrom = self._repair_chromosome(self.base_chromosome + noise)
            population.append(chrom)
        return population

    def _build_temporary_engine(self, chromosome):
        """Builds a temporary Fuzzy Inference System using the chromosome's parameters."""
        variables = define_variables()
        S = variables['suitability']
        
        # Override specific membership functions with chromosome genes
        S['Low'] = fuzz.trimf(S.universe, [chromosome[0], chromosome[1], chromosome[2]])
        S['Moderate'] = fuzz.trimf(S.universe, [chromosome[3], chromosome[4], chromosome[5]])
        S['High'] = fuzz.trimf(S.universe, [chromosome[6], chromosome[7], chromosome[8]])
        
        rules = build_rule_base(variables)
        fis = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(fis)

    def _evaluate_fitness(self, chromosome):
        """
        Fitness Function: Average NDCG@3 across all evaluation scenarios.
        """
        simulator = self._build_temporary_engine(chromosome)
        k = 3
        ndcg_total = 0.0
        
        for scenario in self.scenarios:
            results = []
            for r in scenario['restaurants']:
                feats = r['features']
                
                # Fuzzification and inference
                simulator.input['rating'] = np.clip(feats['rating'], 0, 5)
                simulator.input['price_suitability'] = np.clip(feats['price_suit'], 0, 100)
                simulator.input['distance'] = np.clip(feats['dist'], 0, 15)
                simulator.input['cuisine_match'] = np.clip(feats['cuisine'], 0, 100)
                
                try:
                    simulator.compute()
                    score = simulator.output['suitability']
                except ValueError:
                    score = 0.0 # Penalty for no rule activation
                    
                results.append({'score': score, 'true_relevance': r['true_relevance']})
            
            # Rank descending
            results.sort(key=lambda x: x['score'], reverse=True)
            rels = [x['true_relevance'] for x in results]
            ndcg_total += ndcg_at_k(rels, k)
            
        return ndcg_total / len(self.scenarios)

    def _selection(self, population, fitnesses):
        """Tournament selection."""
        idx1 = np.random.randint(0, self.pop_size)
        idx2 = np.random.randint(0, self.pop_size)
        if fitnesses[idx1] > fitnesses[idx2]:
            return population[idx1]
        return population[idx2]

    def _crossover(self, parent1, parent2):
        """Arithmetic crossover."""
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return self._repair_chromosome(child1), self._repair_chromosome(child2)

    def _mutate(self, chromosome):
        """Gaussian mutation."""
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 5) # mean 0, std 5
        return self._repair_chromosome(chromosome)

    def run(self):
        """Executes the Genetic Algorithm optimization loop."""
        print(f"Starting GA Optimization (Pop: {self.pop_size}, Gen: {self.generations})...")
        population = self._init_population()
        best_overall_chrom = None
        best_overall_fit = -1.0
        
        for gen in range(self.generations):
            # 1. Evaluate fitness
            fitnesses = [self._evaluate_fitness(chrom) for chrom in population]
            
            # Track best
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            if best_fit > best_overall_fit:
                best_overall_fit = best_fit
                best_overall_chrom = population[best_idx]
                
            self.convergence_history.append(best_fit)
            print(f"Generation {gen+1:02d} | Best NDCG@3: {best_fit:.4f}")
            
            # 2. Elitism
            new_population = [best_overall_chrom.copy()]
            
            # 3. Create next generation
            while len(new_population) < self.pop_size:
                p1 = self._selection(population, fitnesses)
                p2 = self._selection(population, fitnesses)
                
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population.extend([c1, c2])
                
            # Truncate if we exceeded pop_size
            population = new_population[:self.pop_size]

        print("\nGA Optimization Complete.")
        print(f"Baseline Chromosome: {np.round(self.base_chromosome, 2)}")
        print(f"Best Found Chromosome: {np.round(best_overall_chrom, 2)}")
        print(f"Best NDCG@3: {best_overall_fit:.4f}")
        
        return best_overall_chrom, self.convergence_history

if __name__ == "__main__":
    ga = FuzzyBiteGA(scenarios_path="../data/evaluation_scenarios.json")
    best_chrom, history = ga.run()