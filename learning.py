

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 array
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False       # True when actions > 200

        rewards = 0
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            rewards += reward

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001


        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if is_training:
        f = open("taxi.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(15000)

    run(10, is_training=False, render=True)
     
# This code implements Q-learning to train an agent on the "Taxi-v3" environment using the Gym library. The Taxi-v3 environment is a classic reinforcement learning task where a taxi agent needs to pick up and drop off passengers in a 5x5 grid. The goal is to use Q-learning to enable the taxi to learn the best actions to maximize the reward. Let's break down the code step-by-step.

# 1. Imports and Function Definition
# gymnasium as gym: Imports the Gymnasium library for creating the "Taxi-v3" environment.
# numpy as np: Imports the NumPy library, used here for creating and manipulating arrays.
# matplotlib.pyplot as plt: Used to plot the cumulative rewards over episodes.
# pickle: Used to save and load the Q-table.
# python
# Copy code
# def run(episodes, is_training=True, render=False):
# This function run accepts episodes (number of episodes to run), is_training (if True, it will train the agent), and render (if True, it will render the environment).
# 2. Creating the Taxi Environment
# python
# Copy code
# env = gym.make('Taxi-v3', render_mode='human' if render else None)
# Initializes the Taxi-v3 environment. If render is True, it will display the environment in a human-readable format.
# 3. Initializing or Loading the Q-Table
# python
# Copy code
# if(is_training):
#     q = np.zeros((env.observation_space.n, env.action_space.n))
# else:
#     f = open('taxi.pkl', 'rb')
#     q = pickle.load(f)
#     f.close()
# If training, initializes the Q-table q as a zero array with dimensions [500 x 6] (500 possible states, 6 actions).
# If not training, loads a previously saved Q-table from a file taxi.pkl.
# 4. Hyperparameters for Q-Learning
# python
# Copy code
# learning_rate_a = 0.9 
# discount_factor_g = 0.9 
# epsilon = 1         
# epsilon_decay_rate = 0.0001 
# rng = np.random.default_rng()   
# learning_rate_a: Controls how much new knowledge overrides old knowledge.
# discount_factor_g: Determines the importance of future rewards.
# epsilon: Initial exploration rate (100% at start).
# epsilon_decay_rate: Rate at which epsilon reduces to shift from exploration to exploitation.
# rng: Random number generator instance for epsilon-greedy action selection.
# 5. Training Loop
# python
# Copy code
# rewards_per_episode = np.zeros(episodes)
# Initializes an array to store rewards for each episode.
# Episode Loop
# python
# Copy code
# for i in range(episodes):
#     state = env.reset()[0]
#     terminated = False
#     truncated = False
#     rewards = 0
# For each episode, state is reset, and flags terminated and truncated are set to False to indicate if the episode is still active.
# rewards stores the cumulative reward for the current episode.
# Action Selection and Q-Update
# python
# Copy code
# while(not terminated and not truncated):
#     if is_training and rng.random() < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(q[state,:])
# Inside each step, the agent chooses an action based on the epsilon-greedy strategy:
# With probability epsilon, it takes a random action (exploration).
# Otherwise, it selects the best-known action based on the Q-table (exploitation).
# python
# Copy code
# new_state, reward, terminated, truncated, _ = env.step(action)
# rewards += reward
# Executes the chosen action and receives:
# new_state: The next state after the action.
# reward: The reward for the action.
# terminated and truncated: Flags indicating if the episode is over.
# python
# Copy code
# if is_training:
#     q[state, action] = q[state, action] + learning_rate_a * (
#         reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action]
#     )
# Q-learning Update:

# The Q-value for the (state, action) pair is updated using the Bellman equation.
# It considers the current reward and the maximum future reward from new_state.
# state = new_state updates the state for the next step in the episode.

# Epsilon Decay and Learning Rate Adjustment
# python
# Copy code
# epsilon = max(epsilon - epsilon_decay_rate, 0)
# if(epsilon == 0):
#     learning_rate_a = 0.0001
# epsilon decays over time, reducing exploration as training progresses.
# When epsilon reaches 0, the learning_rate_a is significantly reduced to stabilize learning.
# Episode Reward Storage
# python
# Copy code
# rewards_per_episode[i] = rewards
# Stores the total reward for each episode in rewards_per_episode.
# 6. Plotting Rewards Over Episodes
# python
# Copy code
# sum_rewards = np.zeros(episodes)
# for t in range(episodes):
#     sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
# plt.plot(sum_rewards)
# plt.savefig('taxi.png')
# Calculates a moving sum of rewards over the last 100 episodes to smooth out reward progress.
# Plots and saves the cumulative rewards as taxi.png.
# 7. Saving the Q-Table
# python
# Copy code
# if is_training:
#     f = open("taxi.pkl", "wb")
#     pickle.dump(q, f)
#     f.close()
# Saves the trained Q-table to taxi.pkl if training is enabled, so it can be loaded later.
# 8. Running the Code in Main Block
# python
# Copy code
# if __name__ == '__main__':
#     run(15000)
#     run(10, is_training=False, render=True)
# Runs the training phase for 15,000 episodes.
# Then, runs 10 episodes in non-training mode with render=True to visualize the agent’s performance.
# Summary of How It Works:
# Training Phase: The agent explores different actions in the Taxi environment using Q-learning to update a Q-table that represents the expected rewards for each (state, action) pair.
# Testing Phase: Once trained, the agent can be tested by loading the Q-table and seeing how it performs without further updates.     




# Q-learning is a type of reinforcement learning (RL), where an agent learns to make decisions by interacting with an environment. By trial and error, it finds the actions that maximize the rewards it gets over time.

# Core Concepts in Q-learning
# To understand how Q-learning works, let’s go through its essential components and then dive into the code.

# Environment: The space where the agent operates. It has states (conditions or situations) and rewards (feedback) for actions the agent takes. In this case, the environment is the Taxi-v3 environment from the Gym library, where a taxi agent must pick up and drop off passengers.

# Agent: This is the decision-making entity in RL (the taxi in this example). The agent’s goal is to maximize its cumulative reward by learning which actions yield the most reward in the long run.

# State: A unique situation or configuration within the environment. In Taxi-v3, each possible position and passenger/drop-off combination is a state.

# Action: A move the agent can make. Actions in the Taxi-v3 environment are to move up, down, left, right, or to pick up/drop off a passenger.

# Reward: Feedback the agent gets after taking an action in a state. The reward guides the agent toward the goal, either penalizing or rewarding it for certain actions.

# Q-table: A table that maps states to action values (expected rewards for taking each action from that state). Each entry in the Q-table represents the agent's "belief" in how good a particular action is in a specific state.

# How Q-learning Works
# Q-learning is all about trial and error and gradually improving the Q-table based on the agent's experiences. Let’s break down the steps of this learning process:

# Initialize Q-table: At the beginning, the agent has no knowledge of the environment, so it initializes a Q-table (basically a big lookup table) where all entries are zero.

# Exploration vs. Exploitation: The agent chooses actions based on an epsilon-greedy strategy:

# With a probability epsilon, it picks a random action to explore the environment (exploration).
# Otherwise, it picks the action with the highest value in the Q-table for its current state (exploitation).
# Over time, the agent decreases epsilon to rely more on what it has learned.

# Update Q-values:

# After taking an action, the agent receives a reward and observes the next state.

# It then updates its Q-value for the (state, action) pair using the Q-learning update rule:

# 𝑄 (𝑠,𝑎 ) = 𝑄( 𝑠,𝑎 )+ 𝛼 ×(reward+𝛾× max(𝑄(new_state,:))−𝑄(𝑠,𝑎))
# Q(s,a)=Q(s,a)+α×(reward+γ×max(Q(new_state,:))−Q(s,a))
# where:

# 𝛼
# α (learning rate) controls how quickly the agent updates its knowledge.
# 𝛾
# γ (discount factor) balances the importance of immediate vs. future rewards.
# Repeat for Multiple Episodes: By iterating through episodes, the agent explores more states, receives feedback (rewards), and gradually learns which actions yield the best cumulative rewards over time.

# Testing Phase: After training, the agent loads the Q-table and performs actions based on what it has learned, without updating the Q-table.

# Real-World Examples of Q-learning
# Q-learning is particularly useful when we don’t have a clear model of the environment but can observe outcomes based on actions taken. Here are some practical applications:

# Robot Navigation:

# Imagine a cleaning robot in a warehouse. It needs to learn how to navigate around obstacles and avoid hazardous areas while cleaning efficiently. Using Q-learning, the robot could learn which paths maximize its cleaning efficiency and minimize risk.
# Autonomous Vehicles:

# Self-driving cars face continuous decision-making challenges: avoiding obstacles, changing lanes, and handling traffic lights. Using Q-learning, the car could learn to make safer decisions by interacting with simulated traffic environments.
# Game Playing:

# Q-learning is widely used in games. For example, an AI agent could learn to play a game like Pong by trial and error, improving over time as it figures out which paddle movements maximize its score.
# Energy Management in Smart Grids:

# In smart grids, managing energy efficiently is critical. Q-learning can help an agent decide how to allocate or conserve energy based on usage patterns, weather conditions, and grid load, balancing immediate and future energy needs.
# Financial Portfolio Management:

# In finance, Q-learning could help manage a portfolio by learning which asset allocations yield the highest returns over time based on changing market conditions.
# Why Q-learning Works Well
# Q-learning is effective in situations where:

# The environment is partially observable or unknown.
# Immediate rewards and future rewards need to be balanced.
# The agent can gradually learn from mistakes to improve its policy.
# In this code, Q-learning allows the agent (taxi) to try picking up and dropping off passengers through random exploration. As it receives feedback (positive or negative rewards), it builds a policy by adjusting values in its Q-table, making it progressively more likely to choose actions that lead to higher rewards. After training, the taxi agent can navigate effectively based on what it has learned from interacting with the environment.

# Summary of the Process
# Start with an empty Q-table.
# Take actions based on exploration (random) or exploitation (best known action).
# Observe rewards and update Q-values accordingly.
# Repeat this across multiple episodes, allowing the agent to learn the best policies through trial and error.
# After training, the agent uses the learned Q-table to make optimal decisions based on its experience.