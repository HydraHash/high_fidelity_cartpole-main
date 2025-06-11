import gym
import numpy as np
import os
import random
import cartpole_realistic

EPISODES = 10
MAX_STEPS = 8

env = gym.make("cartpole-realistic", evaluation=True, swingup=False)

states = []
actions =[]
next_states = []

sequences =  [
    [4, 4, 4, 4], 
    [3, 4, 4, 4], [3, 3, 4, 4], [5, 4, 4, 4], [5, 5, 4, 4],
    [3, 4, 5, 4], [3, 3, 5, 4], [5, 4, 3, 4], [5, 5, 3, 4]
]

def get_action_patterns():
    action_array = []
    for _ in range(int(MAX_STEPS / 4)):
        n = random.randint(0, 8)
        action_array.extend(sequences[n])
    return action_array


for episode in range(EPISODES):
    obs = env.reset()

    #Set pole upright
    env.state[2] = 0.0
    env.state[3] = 0.0
    obs = np.array(env.state, dtype = np.float32)

    #Select actions to perform
    pattern = get_action_patterns()

    for step in range(MAX_STEPS):
        action = pattern[step]
        current_state = np.array(env.state, dtype=np.float32)

        next_obs, reward, done, info = env.step(action)
        next_state = np.array(env.state, dtype=np.float32)

        states.append(current_state)
        actions.append([action])
        next_states.append(next_state)

        if done:
            break

#Save dataset
states = np.array(states)
actions = np.array(actions)
next_states = np.array(next_states)

np.savez("gp_training_upright.npz", states=states, actions=actions, next_states=next_states)
print(f"Saved dataset with {states.shape[0]} samples to gp_training_upright.npz")