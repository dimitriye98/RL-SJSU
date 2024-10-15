from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging

env_logger = logging.getLogger("environmnet_logger")

class Agent(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.runs = [[]]
    @abstractmethod
    def act(self, observation, periphral=None):
        pass
    @abstractmethod
    def think(self, observation_old, action, reward, observation, terminated):
        pass

def run_upto_n_steps(env, agent: Agent, n, continuation=None, runs=[[]]):
    """
    Run n steps in environment or untill termination
    """
    if continuation is not None:
        observation, reward, terminated, truncated, info = continuation
    if continuation is None or terminated or truncated:
        env_logger.info("Resetting")
        observation, info = env.reset()
        terminated = False
        truncated = False
        reward = 0
    step = 0
    # print((observation, reward, terminated, truncated, info))
    while not terminated and not truncated and step < n:
        action = agent.act(observation)
        observation_old = observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.think(observation_old, action, reward, observation, terminated)
        runs[-1].append(reward)
        step += 1
    if terminated or truncated:
        env_logger.info(f"Finished episode with {len(runs[-1])} steps")
        runs.append([])
    return (observation, reward, terminated, truncated, info), runs

def plot_reward_and_episodes(runs):
    # plt.clf()
    plt.scatter(*zip(*enumerate([sum(i) for i in runs])), s=10, c='tab:blue', label='total reward')
    plt.axhline(0, linestyle=':', color='tab:blue')
    plt.ylabel("Total episode reward")
    plt.legend()
    plt.twinx()
    plt.scatter(*zip(*enumerate([len(i) for i in runs])), s=10, c='tab:orange', label='episode length')
    plt.axhline(0, linestyle=':', color='tab:orange')
    plt.ylabel("Episode lenght")
    plt.legend()
    plt.pause(0.1)


def run_and_plot(env, agent: Agent, n, cont=None, ):
    """
    Run and plot agent performance
    """
    cont = run_upto_n_steps(env, agent, 10, cont)
    return cont


import torch.nn as nn
import torch
import numpy as np
class LanderNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
    def forward(self, x):
        return self.layers(x)
class TrainableNetworkAgent(Agent):
    def __init__(self, gamma = 0.9, lr=0.001, update_interval=32, epsilon=0.1) -> None:
        super().__init__()
        self.network = LanderNetwork()
        self.gamma = gamma
        self.update_interval = update_interval
        self.epsilon = epsilon
        self.optim = torch.optim.AdamW(self.network.parameters(), lr)
        self.loss = torch.tensor(0., requires_grad=True)
        self.thoughts = 0
        self.losses = []
        self.events = [] # List of dictionary of events
    def act(self, observation, periphral=None):
        if self.network.training and torch.rand((1,)).item() < self.epsilon:
            return torch.randint(self.network.layers[-1].out_features, (1,)).item()
        obs = torch.from_numpy(observation)
        with torch.no_grad():
            action = self.network(obs)
        return action.argmax().item()
    def think(self, observation, action, reward, observation_next, terminated):
        if not self.network.training:
            return 
        self.thoughts += 1
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        if isinstance(observation_next, np.ndarray):
            observation_next = torch.from_numpy(observation_next)
        outputs = self.network(observation)
        # next_outputs = self.network(observation_next)
        cr_pred = outputs[action]
        cr_esti = reward + (self.gamma*self.network(observation_next).argmax() if not terminated else 0)

        loss = (cr_esti - cr_pred)**2
        # if terminated:
        #     loss = loss + self.terminal_multiplier*sum(next_outputs**2)
        self.loss = self.loss + loss
        self.losses.append(loss.item())
        self.events.append({
            "terminal": terminated,
            "reward": reward
        })
        if self.thoughts % self.update_interval == 0:
            self.update()
        
    def update(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()
        self.loss = torch.tensor(0., requires_grad=True)