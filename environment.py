import os
import random
import torch
import numpy as np 
import gym_super_mario_bros
import yaml
import torch.nn as nn
import torch.optim as optim
import torch.transforms as transform
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation 


if torch.backend.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human" if render else "rgb_array")
        
env = JoypadSpace(env, [["right"], ["right", "A"]])

class SkipFrame:
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, truncate, info = self.env.step(action)
            total_reward += reward
            if done or truncate:
                break
        return obs, total_reward, done, truncate, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape[0],
            dtype=np.uint8
        ) 

    def permute_orientation(self, observation):
        # Convert the [H, W, C]array to [C, H, W]    
        observation =  np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        observation = T.Grayscale()
        observation = transform(observation)
        return observation