from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from train_gp import ExactGPModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

import tensorflow as tf
import cartpole_realistic
import gym
import torch
import gpytorch

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


from tensorflow.python.client import device_lib

def preprocess(path="gp_training_upright.npz"):
    data = np.load(path)
    states = data["states"]
    actions = data["actions"].ravel()

    print(f"Number of actions: {len(actions)}")
    print(f"Raw states shape: {states.shape}")
    print(f"Raw actions shape: {actions.shape}")

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    ) 

env_name = "cartpole-realistic" # @param {type:"string"}

eval_gym_env = gym.make(env_name,evaluation=True)
eval_py_env = suite_gym.wrap_env(eval_gym_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print('load state')

checkpoint = torch.load("gp_policy_model.pt")
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x, train_y = preprocess()
model = ExactGPModel(train_x, train_y, likelihood)
model.load_state_dict(checkpoint["model_state_dict"])
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

model.eval()
likelihood.eval()

print('start gen video')

def gp_policy(state):
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    print(f"Input to model: {x.shape}")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
        mean = observed_pred.mean
        
        return int(round(mean.item()))


num_episodes = 10
video_filename = 'pendulum.mp4'
with imageio.get_writer(video_filename, fps=50) as video:
  for i in range(num_episodes):
    print('episode: ', i)
    time_step = eval_env.reset()
    eval_gym_env.state[2] = 0.0
    eval_gym_env.state[3] = 0.0
    old_frame = None
    video.append_data(eval_py_env.render())
    i = 0
    while not time_step.is_last() and i < 1000:
        obs = time_step.observation.numpy()[0]
        action_step = gp_policy(obs)
        print("Action step: ", action_step)
        time_step = eval_env.step(action_step)
        print("Observation:", time_step.observation.numpy()[0])
        
        frame = eval_py_env.render()
        video.append_data(frame)

        old_frame = frame
        i += 1
