import gymnasium as gym
import numpy as np
import torch as th

env_str = "CartPole-v1"
state_space = gym.make(env_str).observation_space
action_space = gym.make(env_str).action_space


def eval(qtable):
  env = gym.make(env_str, render_mode="human")
  obs, _ = env.reset()
  obs = th.tensor(obs, dtype=th.float32)
  for step in range(100):
    action = act(obs, qtable, eps=0)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
      obs, _ = env.reset()
    obs = th.tensor(obs, dtype=th.float32)

def act(obs, qmodel, eps=0.2):
  if np.random.random(1) < eps:
    return action_space.sample()
  action_values = np.zeros(action_space.n)
  for action_i in range(action_space.n):
    action = th.nn.functional.one_hot(th.tensor(action_i), num_classes=action_space.n).to(th.float32)
    action_values[action_i] = qmodel(th.cat([obs, action], dim=0))
  return np.argmax(action_values).astype(int)


def qmodel_forward(qmodel, obs, no_grad=False):
  if no_grad:
    with th.no_grad():
      action = th.tensor(act(obs, qmodel, eps=0))
      action_oh = th.nn.functional.one_hot(action, num_classes=action_space.n).to(th.float32)
      return qmodel(th.cat([obs, action_oh], dim=0))
  else:
    action = th.tensor(act(obs, qmodel, eps=0))
    action_oh = th.nn.functional.one_hot(action, num_classes=action_space.n).to(th.float32)
    return qmodel(th.cat([obs, action_oh], dim=0))


def train(n_steps=100, eps=1.0, learning_rate=0.1):
  qmodel = th.nn.Sequential(
    th.nn.Linear(state_space.shape[0] + action_space.n, 128), th.nn.ReLU(),
    th.nn.Linear(128, 128), th.nn.ReLU(),
    th.nn.Linear(128, 1)
  )
  loss_fn = th.nn.MSELoss()
  optimizer = th.optim.Adam(qmodel.parameters(), lr=learning_rate)

  env = gym.make(env_str)
  obs, _ = env.reset()
  obs = th.tensor(obs, dtype=th.float32)
  tot_reward = 0
  for step in range(n_steps):
    action = act(obs, qmodel, eps)
    new_obs, reward, terminated, truncated, info = env.step(action)
    # turn everything into tensors
    reward = th.tensor(reward, dtype=th.float32)
    new_obs = th.tensor(new_obs, dtype=th.float32)
    # calculate target and predicted values
    q_target_value = reward + 0.99 * qmodel_forward(qmodel, new_obs, no_grad=True)
    q_pred_value = qmodel_forward(qmodel, obs)
    # calculate loss
    loss = (q_target_value - q_pred_value) ** 2
    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tot_reward += reward
    if terminated:
      new_obs, _ = env.reset()
      new_obs = th.tensor(new_obs, dtype=th.float32)
    if step % 100 == 0:
      print(f"Step {step}: {loss.item()}")
    obs = new_obs
  return qmodel


qmodel = train(10000, eps=1.0, learning_rate=0.1)

eval(qmodel)

import matplotlib.pyplot as plt
# sample and plot the learned q-values for an environment
env = gym.make(env_str)
observations = [th.tensor(obs, dtype=th.float32) for obs in env.observation_space.sample(512)]
qvalues = [qmodel_forward(qmodel, obs, no_grad=True) for obs in observations]
x_axis_idx = 0
y_axis_idx = 2
plt.scatter([obs[x_axis_idx].item() for obs in observations], [obs[y_axis_idx].item() for obs in observations], c=[qvalue.item() for qvalue in qvalues])
plt.show()
eval(qtable)