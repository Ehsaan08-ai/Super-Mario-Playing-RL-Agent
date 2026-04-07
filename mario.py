import argparse
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class MarioAgent:
    def __init__(self, state_dim, action_dim, lr=0.00025, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=1e-5, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        self.net = MarioNet(state_dim, action_dim).float().to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(state_dim, action_dim)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(q_values).item()

        action = np.zeros(self.action_dim)
        action[action_idx] = 1.0
        return action
    
    def learn(self, batch_size=32):
        if self.buffer.size < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        q_values = self.net(states, model='online')
        current_q = torch.sum(q_values * actions, axis=1, keepdim=True)

        next_q_online = self.net(next_states, model='online')
        best_actions = torch.argmax(next_q_online, axis=1, keepdim=True)

        next_q_target = self.net(next_states, model='target')
        next_q_value = torch.gather(next_q_target, 1, best_actions)

        target_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for online_params, target_params in zip(self.net.online.parameters(), self.net.target.parameters()):
            target_params.data.copy_(self.tau * online_params.data + (1.0 - self.tau) * target_params.data)

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return loss.item()

def train(num_episodes=500):
    env = create_env()
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    agent = MarioAgent(state_dim, action_dim)

    best_reward = -float('inf')

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_loss = 0.0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(np.argmax(action))

            terminal = done or info.get('flag_get', False)

            agent.buffer.store(state, action, reward, next_state, float(terminal))

            loss = agent.learn(batch_size=32)
            ep_loss += loss

            state = next_state
            ep_reward += reward
            steps += 1

            if terminal:
                break

        avg_loss = ep_loss / steps if steps > 0 else 0
        print(f"Ep: {ep+1}/{num_episodes} | Reward: {ep_reward:>6.1f} | Steps: {steps:>4} | Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.net.online.state_dict(), "best_mario_model.pth")
            print(f"New best model saved! (Reward: {best_reward})")

    torch.save(agent.net.online.state_dict(), "final_mario.pth")
    env.close()

# Testing
def test(model_path="best_mario_pth", num_episodes=3):
    if not os.path.exists(model_path):
        print(f"Error: could not find '{model_path}'. You  must train first!")
        return

    env = create_env()

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # Initialize Agent
    agent = MarioAgent(state_dim, action_dim)

    # Load the saved brain into the online network. 
    agent.net.online.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.net.online.eval()

    agent.epsilon = 0.0 # Set epsilon to 0 so mario only uses his trained brain (no random choice)

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        info = {}

        while True:
            env.render()

            # Choose action purely from trained NN
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(np.argmax(action))

            state = next_state
            total_reward += reward

            time.sleep(0.01)
            if done or info.get('flag_get', False):
                break

        print(f"Episode finished. Total Reward: {total_reward} | Distance (X-Pos): {info.get('x_pos', 0)}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario DQN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Use 'train' to train without seeing, 'test' to watch AI Play")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train or test.")
    parser.add_argument("--model", type=str, default="best_mario_model.pth", help="Which saved model file to use for testing.")

    args = parser.parse_args()

    if args.mode == "train":
        train(num_episodes=args.episodes)
    else: 
        test(model_path=args.model, num_episodes=args.episodes)