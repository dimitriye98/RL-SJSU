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

def run_upto_n_steps(env, agent: Agent, n, continuation=None):
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
        agent.runs[-1].append(reward)
        agent.think(observation_old, action, reward, observation, terminated)
        step += 1
    if terminated or truncated:
        env_logger.info(f"Finished episode with {len(agent.runs[-1])} steps")
        agent.runs.append([])
    return (observation, reward, terminated, truncated, info)

def plot_reward_and_episodes(agent: Agent):
    # plt.clf()
    plt.scatter(*zip(*enumerate([sum(i) for i in agent.runs])), s=10, c='tab:blue', label='total reward')
    plt.axhline(0, linestyle=':', color='tab:blue')
    plt.ylabel("Total episode reward")
    plt.legend()
    plt.twinx()
    plt.scatter(*zip(*enumerate([len(i) for i in agent.runs])), s=10, c='tab:orange', label='episode length')
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
