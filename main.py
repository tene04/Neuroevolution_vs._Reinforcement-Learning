import gymnasium as gym
import flappy_bird_gymnasium 
import pickle
import torch
import time

from NE.train_neo import * 
from DQL.train_dql import *


env_lidar = gym.make("FlappyBird-v0", use_lidar=True) 
obs_size_lidar = env_lidar.observation_space.shape[0]
env = gym.make("FlappyBird-v0", use_lidar=False) 
obs_size = env.observation_space.shape[0]
n_actions = env_lidar.action_space.n
device = 'cuda' if torch.cuda.is_available() else 'cpu'  


# print("Training neuroevolution agent...")
# best_neo_lidar, max_fitness_lidar, avg_fitness_lidar = neuroevolution_learn(
#     input=obs_size_lidar,
#     hidden=[256, 128],
#     output=n_actions,
#     elite_fraction=0.2,
#     pmut=0.15,
#     n_gen=300,
#     trace=30,
#     lidar=True
# )

# with open('./data/neo_lidar.pkl', 'wb') as f:
#     pickle.dump((best_neo_lidar, max_fitness_lidar, avg_fitness_lidar), f)


# print("Training neuroevolution agent without lidar...") 
# best_neo, max_fitness, avg_fitness = neuroevolution_learn(
#     input=obs_size,
#     hidden=[256, 64],
#     output=n_actions,
#     elite_fraction=0.2,
#     pmut=0.15,
#     n_gen=800,
#     trace=80,
#     lidar=False
# )

# with open('./data/neo.pkl', 'wb') as f:
#     pickle.dump((best_neo, max_fitness, avg_fitness), f)


print("Training deep Q-Learning agent...")
a = time.time()
best_dql_lidar, rewards_per_episode_lidar, epsilon_history_lidar = dql_learning(
    input=obs_size_lidar,
    hidden=[256, 128],
    output=n_actions,
    eps=0.1,
    eps_decay=0.99995,
    eps_min=0.00001,
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    nsr=10,
    memory_max_len=150000,
    device=device,
    env=env_lidar,
    episodes=80000,
    trace=8000
)
b = time.time()

print((b-a) / 60)

with open('./data/dql_lidar.pkl', 'wb') as f:
    pickle.dump((best_dql_lidar, rewards_per_episode_lidar, epsilon_history_lidar), f)


# print("Training deep Q-Learning agent without lidar...")
# best_dql, rewards_per_episode, epsilon_history = dql_learning(
#     input=obs_size,
#     hidden=[256, 64],
#     output=n_actions,
#     eps=0.1,
#     eps_decay=0.99995,
#     eps_min=0.00001,
#     gamma=0.99,
#     lr=1e-4,
#     batch_size=128,
#     nsr=10,
#     memory_max_len=150000,
#     device=device,
#     env=env,
#     episodes=100000,
#     trace=10000
# )

# with open('./data/dql.pkl', 'wb') as f:
#     pickle.dump((best_dql, rewards_per_episode, epsilon_history), f)


print("Training completed and results saved to .data directory.")