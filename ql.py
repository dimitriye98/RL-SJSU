# Importing necessary libraries
import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Initialize Frozen Lake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Q-learning hyperparameters
learning_rate = 0.8     # Alpha: How much we trust new Q-values over old ones
discount_factor = 0.95  # Gamma: How much we value future rewards over immediate rewards
exploration_rate = 1.0  # Epsilon: Starting exploration rate
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001  # Decay of epsilon after each episode
num_episodes = 10000    # Total training episodes
max_steps_per_episode = 100  # Max steps in one episode

# Initialize Q-table with zeros (State x Action table)
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))

# List of rewards for plotting
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # Reset environment
    state, _ = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])  # Exploit: Choose action with highest Q-value
        else:
            action = env.action_space.sample()  # Explore: Random action

        # Take the action in the environment and observe new state, reward, and done flag
        new_state, reward, done, _, _ = env.step(action)

        # Update Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

        # Transition to new state
        state = new_state

        rewards_current_episode += reward
        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Store rewards of each episode
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and plot average rewards per 1000 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
for r in rewards_per_thousand_episodes:
    print(f"Average reward in episodes {count-1000} to {count}: {sum(r) / 1000}")
    count += 1000

# Plotting rewards
plt.plot(range(num_episodes), rewards_all_episodes)
plt.title("Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
# Test the agent after training
for episode in range(3):
    state, _ = env.reset()
    done = False
    print(f"***EPISODE {episode+1}***\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        env.render()
        action = np.argmax(q_table[state,:])  # Exploit learned policy
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        if done:
            if reward == 1:
                print("Reached the goal!")
            else:
                print("Fell into a hole.")
            break
        time.sleep(0.05)  # Slow down for visualization

env.close()
