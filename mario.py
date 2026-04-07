import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
import cv2
from collections import deque
import random
import time
import os
import argparse

# ==========================================
# 1. OPTIMIZED ENVIRONMENT WRAPPERS
# ==========================================

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs_buffer = [] 
        
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
                
        if len(obs_buffer) >= 2:
            max_frame = np.max(np.stack(obs_buffer[-2:]), axis=0)
        else:
            max_frame = obs_buffer[0] 
            
        return max_frame, total_reward, done, info

class FrameStack(gym.Wrapper):
    def __init__(self, env, n=4):
        super().__init__(env)
        self.n = n
        self.frames = deque([], maxlen=n)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n, shp[0], shp[1]), dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n):
            self.frames.append(obs)
        return np.array(self.frames, dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames, dtype=np.uint8), reward, done, info

def apply_wrappers(env, shape=(84, 84), skip=4, stack=4):
    env = JoypadSpace(env, SIMPLE_MOVEMENT) 
    env = SkipFrame(env, skip=skip)
    env = GrayScaleResize(env, shape=shape)
    env = FrameStack(env, n=stack)
    return env

# ==========================================
# 2. THE NEURAL NETWORK
# ==========================================

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
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
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.online[-1].out_features)
        )
        return target

    def forward(self, x, model="online"):
        if model == "online":
            return self.online(x)
        return self.target(x)

# ==========================================
# 3. OPTIMIZED REPLAY BUFFER
# ==========================================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
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
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        return (
            torch.tensor(self.state[idxs], dtype=torch.float32).to('cpu'),
            torch.tensor(self.action[idxs], dtype=torch.float32).to('cpu'),
            torch.tensor(self.reward[idxs], dtype=torch.float32).to('cpu'),
            torch.tensor(self.next_state[idxs], dtype=torch.float32).to('cpu'),
            torch.tensor(self.done[idxs], dtype=torch.float32).to('cpu')
        )

# ==========================================
# 4. THE DOUBLE DQN AGENT
# ==========================================

class MarioAgent:
    def __init__(self, state_dim, action_dim, lr=0.00025, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=1e-5, tau=0.005):
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

        q_values = self.net(states, model="online")
        current_q = torch.sum(q_values * actions, axis=1, keepdim=True)

        next_q_online = self.net(next_states, model="online")
        best_actions = torch.argmax(next_q_online, axis=1, keepdim=True)
        
        next_q_target = self.net(next_states, model="target")
        next_q_value = torch.gather(next_q_target, 1, best_actions)

        target_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0) 
        self.optimizer.step()

        for online_params, target_params in zip(self.net.online.parameters(), self.net.target.parameters()):
            target_params.data.copy_(self.tau * online_params.data + (1.0 - self.tau) * target_params.data)

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return loss.item()

# ==========================================
# 5. TRAINING LOOP (HEADLESS)
# ==========================================

def train(num_episodes=500):
    print("=== STARTING TRAINING (HEADLESS) ===")
    env = gym_super_mario_bros.make('SuperMarioBros2-v0')
    env = apply_wrappers(env)
    
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
            torch.save(agent.net.online.state_dict(), "best_mario.pth")
            print(f"  -> New best model saved! (Reward: {best_reward})")
            
    torch.save(agent.net.online.state_dict(), "final_mario.pth")
    print("=== TRAINING COMPLETE. Models saved as 'best_mario.pth' and 'final_mario.pth' ===")
    env.close()

# ==========================================
# 6. TESTING / PLAYBACK LOOP (VISUAL)
# ==========================================

def test(model_path="best_mario.pth", num_episodes=3):
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. You must train first!")
        return

    print(f"=== LOADING MODEL: {model_path} ===")
    env = gym_super_mario_bros.make('SuperMarioBros2-v0')
    env = apply_wrappers(env)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    agent = MarioAgent(state_dim, action_dim)
    agent.net.online.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.net.online.eval() 
    agent.epsilon = 0.0 

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        info = {}
        
        print(f"\n--- Watching Episode {ep+1} ---")

        while True:
            env.render() 
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(np.argmax(action))
            
            state = next_state
            total_reward += reward
            time.sleep(0.01) 
            
            if done or info.get('flag_get', False):
                break
                
        print(f"Episode finished. Total Reward: {total_reward} | Distance (X-Pos): {info.get('x_pos', 0)}")

    env.close()
    print("=== DEMO FINISHED ===")

# ==========================================
# 7. EXECUTION CONTROL
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario DDQN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model", type=str, default="best_mario.pth")
    args = parser.parse_args()

    if args.mode == "train":
        train(num_episodes=args.episodes)
    else:
        test(model_path=args.model, num_episodes=args.episodes)