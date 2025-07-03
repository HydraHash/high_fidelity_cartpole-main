from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image

import tensorflow as tf
import cartpole_realistic
import gym
import numpy as np
import math

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


from tensorflow.python.client import device_lib

EPISODES = 8
MAX_STEPS = 80
ANGLE_THRESHOLD_DEG = 5
LEFT_ACTIONS = [0,1]
RIGHT_ACTIONS = [7,8]
ALL_ACTIONS = [2,3,4,5,6]

max_total_steps = EPISODES * MAX_STEPS
states = np.empty((max_total_steps, 4), dtype=np.float32)
actions = np.empty((max_total_steps, 1), dtype=int)
rewards = np.empty(max_total_steps, dtype=np.float32)
next_states = np.empty((max_total_steps, 4), dtype=np.float32)
state_index = 0
     

env_name = "cartpole-realistic" # @param {type:"string"}

eval_gym_env = gym.make(env_name,evaluation=True)
eval_py_env = suite_gym.wrap_env(eval_gym_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print('load state')

policy = tf.saved_model.load('policy')

print('start gen video')

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

video_filename = 'pendulum.mp4'
threshold_rad = math.radians(ANGLE_THRESHOLD_DEG)
print("Threshold rad: ", threshold_rad, " negative:  ", -threshold_rad)

with imageio.get_writer(video_filename, fps=50) as video:
    for episode in range(EPISODES):
        print("episode: ", episode)
        obs = eval_gym_env.reset()

        #Set pole upright
        eval_gym_env.state[2] = 0.0
        eval_gym_env.state[3] = 0.0
        obs = np.array(eval_gym_env.state, dtype = np.float32)

        noisy_state = add_state_noise(obs, 0.1)
        obs = noisy_state
        print("Observation noisy state:", obs)
    
        video.append_data(eval_py_env.render())

        for step in range(MAX_STEPS):
            if 2.9 < eval_gym_env.state[2] < 3.2 or -3.2 < eval_gym_env.state[2] < -2.9 or eval_gym_env.state[0] < -0.24 or eval_gym_env.state[0] > 0.24:
                print("break")
                break
            action = get_action_heuristic(eval_gym_env.state[2], threshold_rad)
            current_state = np.array(eval_gym_env.state, dtype=np.float32)
            
            next_obs, reward, done, info = eval_gym_env.step(action)
            next_state = np.array(eval_gym_env.state, dtype=np.float32)

            states[state_index] = current_state
            actions[state_index] = action
            rewards[state_index] = reward
            next_states[state_index] = next_state
            state_index += 1

            #print("Observation:", next_obs)

            frame = eval_py_env.render()
            video.append_data(frame)
            old_frame = frame

#Save dataset
states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
next_states = np.array(next_states)

np.savez("gp_training_upright.npz", states=states, actions=actions, rewards=rewards, next_states=next_states)
print(f"Saved dataset with {states.shape[0]} samples to gp_training_upright.npz")