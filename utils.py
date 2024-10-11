from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, observation, periphral=None):
        pass
    @abstractmethod
    def think(self, observation_old, action, reward, observation, terminated):
        pass

def run_n_steps(env, agent: Agent, n, continuation=None):
    if continuation is not None:
        observation, reward, terminated, truncated, info = continuation
    if continuation is None or terminated or truncated:
        print("Resetting")
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
        step += 1
    return (observation, reward, terminated, truncated, info)