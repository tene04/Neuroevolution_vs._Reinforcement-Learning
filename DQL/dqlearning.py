from collections import deque
import random
from utils.MLP import MLP
import torch
from torch import nn
import numpy as np

class DQLearning():

    def __init__(self, input, hidden, output, eps, eps_decay, eps_min, gamma, lr, batch_size, nsr, memory_max_len, device):
        """
        Initializes the DQLearning agent 

        Args:
            input (int): Dimension of the input state
            hidden ([int]): Sizes of hidden layers
            output (int): Number of possible actions
            eps (float): Initial epsilon for epsilon-greedy exploration
            eps_decay (float): Factor by which epsilon decays
            eps_min (float): Minimum value for epsilon
            gamma (float): Discount factor for future rewards
            lr (float): Learning rate
            batch_size (int): Mini-batch size used in optimization
            nsr (int): Number of steps before updating the target network
            memory_max_len (int): Maximum size of the replay memory
            device (torch.device): Device to run the model on
        """
        self.device = device
        self.input = input
        self.hidden = hidden
        self.output = output
        self.policy = MLP(self.input, self.hidden, self.output).to(self.device)
        self.target = MLP(self.input, self.hidden, self.output).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsr = nsr
        self.memory = deque([], maxlen=memory_max_len)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.step = 0

    def optimize(self, mini_batch):
        """
        Performs a single optimization step using a mini-batch of transitions

        Args:
            mini_batch (list of tuples): Each tuple contains (state, action, next_state, reward, done)
        """
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
    
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(self.device)
    
        with torch.no_grad():
            best_actions = self.policy(new_states).argmax(dim=1)
            target_q = rewards + (1-terminations) * self.gamma * self.target(new_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
                
        current_q = self.policy(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
    
        self.optimizer.zero_grad() 
        loss.backward()          
        self.optimizer.step()

    def learn(self, episodes, trace, env):
        """
        Trains the DQN agent 

        Args:
            episodes (int): Number of training episodes
            trace (int): Interval for printing training progress
            env (gym.Env): The environment to interact with
        """
        rewards_per_episode = []
        epsilon_history = []
        best_model = MLP(self.input, self.hidden, self.output).to(self.device)
        best_reward = -float("inf")

        for episode in range(episodes): 
            state, _ = env.reset()  
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            terminated = False    
            episode_reward = 0.0

            while not terminated: 
                if random.random() < self.eps:
                    action = torch.tensor(random.choice([0, 1]), dtype=torch.int64, device=self.device)
                else:
                    with torch.no_grad():
                        action = self.policy(state.unsqueeze(dim=0)).squeeze().argmax() 

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
        
                new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        
                self.memory.append((state, action, new_state, reward, terminated))
                self.step += 1

                terminated = terminated or truncated
                state = new_state

            rewards_per_episode.append(episode_reward)
            epsilon_history.append(self.eps)

            if episode % trace == 0:
                print(f'Episode {episode} | Max reward: {best_reward} | Mean last 5 reward: {np.mean(rewards_per_episode[-5:])} | eps: {self.eps}')

            if episode_reward > best_reward: 
                best_model.load_state_dict(self.policy.state_dict())
                best_reward = episode_reward

            if len(self.memory) > self.batch_size:
                batch = random.sample(self.memory, self.batch_size)
                self.optimize(batch)
                self.eps = max(self.eps * self.eps_decay, self.eps_min)

            if self.step > self.nsr:
                self.target.load_state_dict(self.policy.state_dict())
                self.step = 0

        return best_model, rewards_per_episode, epsilon_history
           