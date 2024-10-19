import gymnasium as gym
import numpy as np

def eval(qtable):
  env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
  obs, _ = env.reset()
  for step in range(100):
    action = act(obs, qtable, eps=0, verbose=1)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
      obs, _ = env.reset()

def act(obs, qtable, eps=0.2, verbose=0):
  if np.random.random(1) < eps:
    if verbose > 0:
      print("hi")
    return np.random.randint(0, 4)
  return np.argmax(qtable[obs])

qtable = np.random.rand(16, 4)
# eval(qtable)
import matplotlib.pyplot as plt
def train(n_steps=100, eps=1.0, learning_rate=0.1):
  # qtable = np.random.rand(16, 4) * 0.01
  qtable = np.zeros((16, 4))
  env = gym.make("FrozenLake-v1", is_slippery=False)
  obs, _ = env.reset()
  tot_reward = 0
  for step in range(n_steps):
    action = act(obs, qtable, eps)
    new_obs, reward, terminated, truncated, info = env.step(action)
    qtable[obs][action] = (1 - learning_rate) * qtable[obs][action] \
        + learning_rate * (reward \
          + 0.99 * qtable[new_obs][act(new_obs, qtable, eps=0)])

    # time.sleep(0.01)
    tot_reward += reward
    if terminated:
      new_obs, _ = env.reset()
    obs = new_obs
  print(tot_reward)
  return qtable

qtable = train(10000, eps=1.0, learning_rate=0.1)
plt.imshow(qtable.max(axis=1).reshape(4,4))
a_to_arrow = [(-1, 0), (0, 1), (1, 0), (0, -1)]
for a in range(4):
  # if a!=1:
  #   continue
  for i in range(16):
    x_ = i % 4
    y_ = i // 4
    dir = a_to_arrow[a]
    plt.arrow(x_, y_, dir[0], dir[1], alpha=qtable[i][a])
plt.show()
print(qtable)
eval(qtable)