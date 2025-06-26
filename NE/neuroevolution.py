import gymnasium as gym
import numpy as np
from utils.MLP import MLP

class Neuroevolution():

    def __init__(self, input, hidden, output, fitness_func, elite_fraction, pmut, N=100):
        """
        Initialize the genetic algorithm

        Args:
            input (int): Size of input layer
            hidden ([int]): Size of hidden layer
            output (int): Size of output layer
            fitness_func (callable): Function that evaluates agent performance
            elite_fraction (float): Fraction of best agents preserved for next generation
            pmut (float): Mutation probability
            N (int): Population size
        """
        self.input = input
        self.hidden = hidden
        self.output = output
        self.fitness_func = fitness_func
        self.elite_fraction = elite_fraction
        self.pmut = pmut
        self.N = N
        self.poblation = self.create()

    def create(self):
        """
        Create the initial population of agents
        """
        return [MLP(self.input, self.hidden, self.output) for _ in range(self.N)]

    def mutate(self, weights):
        """
        Apply mutation to weights

        Args:
            weights (np.ndarray): Array of weights
        """
        new_weights = weights.copy()
        for i in range(len(new_weights)):
            if np.random.rand() < self.pmut:
                new_weights[i] += np.random.normal(0, abs(new_weights[i]) * 0.5 + 1e-3)
        return new_weights

    def evolution(self, ngen, trace, env_name, lidar=False):
        """
        Run the evolutionary algorithm for a number of generations

        Args:
            ngen (int): Number of generations
            trace (int): Trace interval for logging
            env_name (str): Gym environment name
            lidar (bool): If True, use 'use_lidar=True' in environment creation
        """
        env = gym.make(env_name, use_lidar=True) if lidar else gym.make(env_name, use_lidar=False)
        max_fitness = []
        avg_fitness = []
        best_agent = None
        best_fitness = -float('inf')

        for generation in range(ngen):
            fitness = [self.fitness_func(net, env, 'neuroevolution') for net in self.poblation]
            elite_count = int(self.elite_fraction * self.N)
            elite_indices = np.argsort(fitness)[-elite_count:]
            elites = [self.poblation[i] for i in elite_indices]

            if generation % trace == 0:
                print(f"Generation {generation} | Max fitness: {max(fitness)} | Avg fitness: {np.mean(fitness)}")
            max_fitness.append(max(fitness))
            avg_fitness.append(np.mean(fitness))

            current_best_fitness = max(fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_agent = self.poblation[np.argmax(fitness)]

            new_pobla = []
            for _ in range(self.N):
                parent = np.random.choice(elites)
                child = MLP(self.input, self.hidden, self.output)
                child.set_weights(self.mutate(parent.get_weights()))
                new_pobla.append(child)

            self.poblation = new_pobla

        env.close()
        return best_agent, max_fitness, avg_fitness