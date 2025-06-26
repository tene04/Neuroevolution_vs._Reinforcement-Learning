from .dqlearning import DQLearning

def dql_learning(input, hidden, output, eps, eps_decay, eps_min, gamma, lr, batch_size, nsr, memory_max_len, device, env, episodes, trace):
    """
    Run the neuroevolution learning process using the specified parameters

    Args:
        input (int): Dimension of the observation/state space
        hidden (list): List specifying units per hidden layer 
        output (int): Number of possible actions (action space dimension)
        eps (float): Initial epsilon value for ε-greedy exploration
        eps_decay (float): Epsilon decay factor (e.g., 0.999)
        eps_min (float): Minimum epsilon value
        gamma (float): Discount factor for future rewards
        lr (float): Learning rate for the optimizer
        batch_size (int): Batch size for replay memory sampling
        nsr (int): Target network sync frequency (Network Sync Rate)
        memory_max_len (int): Maximum capacity of replay memory
        device (torch.device): Computation device 
        env (gym.Env): The Gym environment to train on
        episodes (int): Number of episode to learn 
        trace (int): When to print progress during training
    
    """
    dql = DQLearning(input, hidden, output, eps, eps_decay, eps_min, gamma, lr, batch_size, nsr, memory_max_len, device)
    best_model, rewards_per_episode, epsilon_history = dql.learn(episodes, trace, env)
    return best_model, rewards_per_episode, epsilon_history




