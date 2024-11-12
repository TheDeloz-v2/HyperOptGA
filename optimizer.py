import numpy as np
import random

class GeneticAlgorithmOptimizer:
    def __init__(self, param_limits, fitness_function, population_size=20, generations=10,
                 crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, verbose=True):
        
        """
        param_limits: dict
            Dictionary with parameter names as keys and tuples (low, high, data_type) as values.
            low: int or float
                Lower bound for the parameter.
            high: int or float
                Upper bound for the parameter.
            data_type: str
                'int' or 'float'.
        fitness_function: function
            Function that takes a dictionary of parameters as input and returns a float score.
        population_size: int
            Number of individuals in the population.
        generations: int
            Number of generations.
        crossover_rate: float
            Probability of crossover.
        mutation_rate: float
            Probability of mutation.
        tournament_size: int
            Number of individuals participating in the tournament.
        verbose: bool
            If True, print messages during optimization.
        """

        self.param_limits = param_limits
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.verbose = verbose

        self.population = []
        self.best_individual = None
        self.best_score = -np.inf
        self.scores = []

    def generate_individual(self):
        """
        Generate a random individual.
        """

        individual = {}
        for param, (low, high, data_type) in self.param_limits.items():
            if data_type == 'int':
                individual[param] = random.randint(low, high)
            elif data_type == 'float':
                individual[param] = random.uniform(low, high)
        return individual

    def generate_population(self):
        """
        Generate the initial population.
        """
        
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def evaluate_population(self):
        """
        Evaluate the population and return a list of scores.
        """

        scores = []
        for individual in self.population:
            score = self.fitness_function(individual)
            scores.append(score)
            if score > self.best_score:
                self.best_score = score
                self.best_individual = individual.copy()
        return scores

    def tournament_selection(self, scores):
        """
        Select individuals for the next generation using tournament selection.
        """

        selected = []
        for _ in range(self.population_size):
            participants = random.sample(list(zip(self.population, scores)), self.tournament_size)
            winner = max(participants, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        """

        child1, child2 = parent1.copy(), parent2.copy()
        for param in self.param_limits.keys():
            if random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate an individual.
        """
        
        for param, (low, high, data_type) in self.param_limits.items():
            if random.random() < self.mutation_rate:
                if data_type == 'int':
                    individual[param] = np.clip(
                        int(np.random.normal(individual[param], 1)),
                        low,
                        high
                    )
                elif data_type == 'float':
                    individual[param] = np.clip(
                        np.random.normal(individual[param], 0.1 * (high - low)),
                        low,
                        high
                    )
        return individual

    def run(self):
        """
        Run the genetic algorithm.
        """

        self.generate_population()
        for generation in range(self.generations):
            if self.verbose:
                print(f'Generation {generation + 1}/{self.generations}')
            scores = self.evaluate_population()
            selected = self.tournament_selection(scores)
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                if i + 1 < self.population_size:
                    parent2 = selected[i + 1]
                else:
                    parent2 = selected[0]
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                self.mutate(child1)
                self.mutate(child2)
                next_population.extend([child1, child2])
            self.population = next_population[:self.population_size]
            if self.verbose:
                print(f'Best score so far: {self.best_score:.4f}')
            self.scores.append(self.best_score)
        return self.best_individual, self.best_score, self.scores