import gym
import numpy as np
import os
import random
import cartpole_realistic
import math

EPISODES = 10000
MAX_STEPS = 400
ANGLE_THRESHOLD_DEG = 40
LEFT_ACTIONS = [2,3]
RIGHT_ACTIONS = [5,6]
ALL_ACTIONS = [0,1,2,3,4,5,6,7,8]

env = gym.make("cartpole-realistic", evaluation=True, swingup=False)

max_total_steps = EPISODES * MAX_STEPS
states = np.empty((max_total_steps, 4), dtype=np.float32)
actions = np.empty((max_total_steps, 1), dtype=int)
rewards = np.empty(max_total_steps, dtype=np.float32)
next_states = np.empty((max_total_steps, 4), dtype=np.float32)
state_index = 0

def get_action_heuristic(angle, threshold):
    if angle < -threshold:
        return np.random.choice(LEFT_ACTIONS)
    elif angle > threshold:
        return np.random.choice(RIGHT_ACTIONS)
    else:
        return np.random.choice(ALL_ACTIONS)

def add_state_noise(state, val=0.01):
    noise = np.random.normal(0, val, size=state.shape)
    noisy_state = state + noise
    return noisy_state

threshold_rad = math.radians(ANGLE_THRESHOLD_DEG)
for episode in range(EPISODES):
    obs = env.reset()

    #Set pole upright
    env.state[2] = 0.0
    env.state[3] = 0.0
    obs = np.array(env.state, dtype = np.float32)

    noisy_state = add_state_noise(obs, 0.1)
    obs = noisy_state

    for step in range(MAX_STEPS):
        action = get_action_heuristic(env.state[2], threshold_rad)
        current_state = np.array(env.state, dtype=np.float32)

        next_obs, reward, done, info = env.step(action)
        next_state = np.array(env.state, dtype=np.float32)

        states[state_index] = current_state
        actions[state_index] = action
        rewards[state_index] = reward
        next_states[state_index] = next_state

        state_index +=1

        if done:
            break

#Save dataset
states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
next_states = np.array(next_states)

np.savez("gp_training_upright.npz", states=states, actions=actions, rewards=rewards, next_states=next_states)
print(f"Saved dataset with {states.shape[0]} samples to gp_training_upright.npz")