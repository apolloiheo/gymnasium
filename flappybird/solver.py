from agent import *
import numpy as np

class Solver:
    def __init__(self, env):
        self.env = env
    
    def solve(self, feature_map: callable, hidden_dim: int, mutation_scale=1e-2,
              population_size: int=10, generations: int=10, reward=Reward,
              initial_pop: int=None, verbose=False):
        '''`feature_map` maps obs to a np.array`'''
        input_dim = len(feature_map((0, 0)))
        output_dim = 1

        # initialize population
        initial_pop = initial_pop or population_size
        population = [
            NeuralNet.create_random(input_dim, hidden_dim, output_dim)
            for _ in range(initial_pop)
        ]

        data = []
        # iterate through generations
        for gen in range(generations+1):
            # collect results
            results = [
                simulate(lambda obs: pop.predict(feature_map(obs)))
                for pop in population
            ]
            fitness = [res[0] for res in results]
            scores = [res[1] for res in results]

            # store results
            data.append({
                'fitness': fitness, 'scores': scores
            })

            # top 2 reproduce
            sorted_idx = np.argsort(fitness)
            nn1, nn2 = population[sorted_idx[-1]], population[sorted_idx[-2]]
            population = nn1.reproduce(nn2, babies_count=population_size,
                                        mutation_scale=mutation_scale)
            
            # print ?
            if verbose and gen % 10 == 0:
                print(f'Gen {gen}:\tBest: {max(scores)}')

        return data, population
