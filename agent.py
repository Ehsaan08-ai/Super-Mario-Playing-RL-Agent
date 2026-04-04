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

class Agent:
    def run(self, is_training=True, render=False):
        env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human" if render else "rgb_array")
        
        