from sympy.printing.pytorch import torch
import tarfile
import imp
import gymnasium as gym 
import torch
import numpy as np 
from collections import deque
import gym_super_mario_bros
import torchvision.transforms as T 
from gymnasium.wrappers import FrameStack
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
import cv2
import random
import time
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip # No. of frames to skip

        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0 # Starting with 0 reward
        for i in range(self._skip): # Loop for the number of frames to skip
            obs, reward, done, truncate, info = self.env.step(action) # This actually moves the mario in the game
            
            # Taking the max of the last 2 frames
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs
            
            total_reward += reward # Adding the reward
            if done or truncate: # If the game is done or truncated
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, truncate, info # Returns the final screen of the skip, the sum of all points


class GrayScalePermutation(gym.ObservationWrapper): # This class turns the game from (RGB) -> Black & White
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2] # Getting the ht. & wt. of the screen.
        self.observation_space = Box( # This line redefines the vision of neural network (from 3 colors to 1 color)
            low=0,
            high=255,
            shape=(1, obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        ) 

    def permute_orientation(self, observation):
        # Convert the [H, W, C]array to [C, H, W]    
        observation =  np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32) # converting the image into tensors the format that the NN understands
        return observation

    def observation(self, observation): # The main fnx that processes the img.
        observation = self.permute_orientation(observation)
        grayscale = T.Grayscale() # This line actually removes the color
        resize = T.Resize((84, 84)) # This line resizes the image
        return resize(grayscale(observation))
    
def create_env(render=False):
    env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human" if render else "rgb_array")       
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, 4) # Skip frames + max pooling
    env = GrayScalePermutation(env) # Grayscale + resize + permute (all-in-one)
    env = FrameStack(env, 4) # Stacking 4 frames together
    return env

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        # online network -> the active brain that plays the game and learns every step
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = self._create_target_network()

    def _create_target_network(self):
        target = nn.Sequential(
            nn.Conv2d(in_channels=self.online[0].in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, self.online[-1].out_features)
        )
        return target

    def forward(self, x, model="online"):
        if model == "online":
            return self.online(x)
        return self.target(x)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size=max_size
        self.ptr=0
        self.size=0

        self.state = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)

        return(
            torch.tensor(self.state[idxs], dtype=torch.float32).to(device),
            torch.tensor(self.action[idxs], dtype=torch.float32).to(device),
            torch.tensor(self.reward[idxs], dtype=torch.float32).to(device),
            torch.tensor(self.next_state[idxs], dtype=torch.float32).to(device),
            torch.tensor(self.done[idxs], dtype=torch.float32).to(device)
        )
