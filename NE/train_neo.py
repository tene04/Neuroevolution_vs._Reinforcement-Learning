from .neuroevolution import Neuroevolution
from utils.test import eval

def neuroevolution_learn(input, hidden, output, elite_fraction, pmut, n_gen, trace, lidar):
    """
    Run the neuroevolution learning process using the specified parameters

    Args:
        input (int): Input size for the MLP
        hidden ([int]): List of hidden layer sizes
        output (int): Output size for the MLP
        elite_fraction (float): Fraction of top-performing individuals to retain
        pmut (float): Probability of mutation
        n_gen (int): Number of generations to evolve
        trace (int): When to print progress during training
        lidar (bool): Whether to use LIDAR input in the environment
    """
    neu = Neuroevolution(input, hidden, output, eval, elite_fraction, pmut)
    best, max_fitness, avg_fitness = neu.evolution(n_gen, trace=trace, env_name="FlappyBird-v0", lidar=lidar)
    return best, max_fitness, avg_fitness
