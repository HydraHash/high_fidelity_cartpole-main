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

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


from tensorflow.python.client import device_lib
     

env_name = "cartpole-realistic" # @param {type:"string"}

eval_gym_env = gym.make(env_name,evaluation=True)
eval_py_env = suite_gym.wrap_env(eval_gym_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print('load state')

policy = tf.saved_model.load('policy')

print('start gen video')


num_episodes = 1
video_filename = 'pendulum.mp4'
with imageio.get_writer(video_filename, fps=50) as video:
  for i in range(num_episodes):
    print('episode: ', i)
    time_step = eval_env.reset()
    old_frame = None
    video.append_data(eval_py_env.render())
    i = 0
    while not time_step.is_last() and i < 1000:
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)

        frame = eval_py_env.render()
        video.append_data(frame)

        old_frame = frame
        i += 1



