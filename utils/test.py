import torch
import numpy as np

def eval(network, env, model, device=None, episodes=5):
    """
    Evaluate the neural network on the environment and return the average reward over several episodes

    Args:
        network: The neural network to evaluate 
        env: The gym-like environment used for evaluation
        model (str): The method used to train
        device (torch.device): Device to run the model on
        episodes (int): Number of episodes to run for averaging
    """
    if model in ['DQL', 'neuroevolution']:
        total_reward = 0
        for _ in range(episodes):
            obs, _ = env.reset()  
            if model == 'DQL':
                obs = torch.tensor(obs, dtype=torch.float, device=device)
            done = False
            while not done: 
                if model == 'DQL':
                    with torch.no_grad():
                        action = network(obs.unsqueeze(dim=0)).squeeze().argmax()
                        new_obs, reward, terminated, truncated, _ = env.step(action.item())
                elif model == 'neuroevolution':
                    action = np.argmax(network.forward(obs, model))
                    new_obs, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward
                done = terminated or truncated

                if model == 'DQL':
                    new_obs = torch.tensor(new_obs, dtype=torch.float, device=device)
                    obs = new_obs
                elif model == 'neuroevolution':
                    obs = new_obs
                
        return total_reward / episodes
    else:
        return 'Not valid model'