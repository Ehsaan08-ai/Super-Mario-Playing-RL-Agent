import os
import gymnasium as gym 
import torch
import gym_super_mario_bros
import torch.nn as nn 
import torch.optim as optim 
import yaml
from environment import create_env 

class Mario:
    env = create_env()
     
    def __init__(self):
        pass

    