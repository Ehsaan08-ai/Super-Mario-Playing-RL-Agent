import os
import random
import torch
import gym_super_mario_bros
import yaml
import torch.nn as nn
import torch.optim as optim
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation 


if torch.backend.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human" if render else "rgb_array")
        
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = ResizeObservation(env, shape=(84, 84))
env = GrayScaleObservation(env)
env = FrameStack(env, num_stack=4)

env.reset()
next_state, reward, done, truncate, info = env.step(action=0)

    