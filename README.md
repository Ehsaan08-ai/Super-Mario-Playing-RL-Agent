# 🍄 Super Mario Playing RL Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Algorithm-Double%20DQN-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-Recommended-orange?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<p align="center">
  A Deep Reinforcement Learning agent that learns to play <strong>Super Mario Bros</strong> using a <strong>Double DQN (DDQN)</strong> architecture with experience replay, frame stacking, and soft target network updates.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Environment](#-environment)
- [GPU Recommendation](#-gpu-recommendation)
- [Setup & Installation](#-setup--installation)
- [Training the Agent](#-training-the-agent)
- [Testing the Agent](#-testing-the-agent)
- [Project Structure](#-project-structure)
- [Hyperparameters](#-hyperparameters)
- [Current Results](#-current-results)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

This project implements a **Double Deep Q-Network (DDQN)** agent that learns to play Super Mario Bros through trial and error — purely from raw pixel observations. The agent observes a preprocessed stack of grayscale frames and outputs the best action to take at each timestep.

### Key Techniques Used

| Technique | Description |
|---|---|
| **Double DQN** | Decouples action selection and evaluation to reduce Q-value overestimation |
| **Experience Replay** | Stores and randomly samples past transitions to break correlation |
| **Frame Stacking** | Stacks 4 consecutive frames to give the agent a sense of motion |
| **Frame Skipping** | Repeats each action for 4 frames to reduce computational cost |
| **Grayscale + Resize** | Converts frames to 84×84 grayscale to reduce input dimensionality |
| **Soft Target Updates** | Slowly syncs target network weights (τ = 0.005) for stable training |
| **Gradient Clipping** | Clips gradients to a max norm of 10.0 to prevent exploding gradients |

---

## 🏗️ Architecture

The agent uses a **Convolutional Neural Network (CNN)** inspired by the DeepMind DQN paper:

```
Input: (4, 84, 84) — 4 stacked grayscale frames

Conv2d(4 → 32, kernel=8, stride=4)  →  ReLU
Conv2d(32 → 64, kernel=4, stride=2) →  ReLU
Conv2d(64 → 64, kernel=3, stride=1) →  ReLU
Flatten
Linear(3136 → 512)  →  ReLU
Linear(512 → num_actions)
```

Two copies of this network are maintained:
- **Online Network** — updated every step (learns from experience)
- **Target Network** — updated slowly via soft update (provides stable Q-targets)

---

## 🎮 Environment

This project uses **`SuperMarioBros2-v0`** from the `gym-super-mario-bros` package.

> ⚠️ **Important Note:** The original `SuperMarioBros-v0` environment is now **deprecated and non-functional** in newer versions of the library. This project uses the fully supported **`SuperMarioBros2-v0`** environment instead.

The action space is simplified using **`SIMPLE_MOVEMENT`** from `nes_py`, reducing the number of possible actions to make learning faster and more stable.

---

## ⚡ GPU Recommendation

> 🔴 **Strongly Recommended: Train on a GPU!**

Training this agent on a **CPU is possible but extremely slow** — a single training run of even 500 episodes can take **several hours** on a typical CPU.

With a **CUDA-compatible NVIDIA GPU**, training the same 500 episodes takes a fraction of that time.

The agent automatically detects and uses CUDA if available:

```python
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Recommended hardware:**
- NVIDIA GTX 1060 or better
- At least 6 GB of VRAM
- CUDA 11.8+ with cuDNN

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.10** (required)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ehsaan08-ai/Super-Mario-Playing-RL-Agent.git
cd Super-Mario-Playing-RL-Agent
```

### Step 2 — Create a Virtual Environment with Python 3.10

> 💡 Using a virtual environment is **strongly recommended** to avoid package conflicts.

```bash
# Create a virtual environment named venv310 using Python 3.10
python3.10 -m venv venv310
```

> If you have multiple Python versions and the above doesn't work, use the full path:
> ```bash
> # Windows example
> "C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv310
> ```

### Step 3 — Activate the Virtual Environment

**Windows (Command Prompt / PowerShell):**
```bash
venv310\Scripts\activate
```

**macOS / Linux:**
```bash
source venv310/bin/activate
```

You should see `(venv310)` at the start of your terminal prompt.

### Step 4 — Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gym==0.26.2
pip install gym-super-mario-bros
pip install nes-py
pip install opencv-python
pip install numpy
```

> 💡 If you don't have a GPU, install the CPU-only version of PyTorch instead:
> ```bash
> pip install torch torchvision torchaudio
> ```

---

## 🏋️ Training the Agent

Make sure your virtual environment is **activated** before running any commands.

### Basic Training (default 500 episodes)

```bash
python mario.py --mode train
```

### Custom Number of Episodes

```bash
python mario.py --mode train --episodes 1000
```

**What happens during training:**
- The agent explores the environment using an ε-greedy strategy
- Experiences are stored in a replay buffer (100,000 transitions max)
- The neural network is updated every step using sampled mini-batches
- The best model is saved as `best_mario.pth` whenever a new high reward is achieved
- The final model is saved as `final_mario.pth` at the end of training

**Training output example:**
```
=== STARTING TRAINING (HEADLESS) ===
Ep: 1/1000 | Reward:  312.0 | Steps:  187 | Eps: 0.998 | Loss: 0.0023
Ep: 2/1000 | Reward:  489.0 | Steps:  256 | Eps: 0.996 | Loss: 0.0041
  -> New best model saved! (Reward: 489.0)
...
```

---

## 🎬 Testing the Agent

Watch the trained agent play Mario in a visual window:

### Using the Best Saved Model (default)

```bash
python mario.py --mode test
```

### Using a Specific Model File

```bash
python mario.py --mode test --model best_mario.pth
```

### Custom Number of Test Episodes

```bash
python mario.py --mode test --model best_mario.pth --episodes 5
```

During testing:
- Epsilon is set to `0.0` — the agent acts **greedily** (no random exploration)
- The environment is rendered visually so you can watch Mario play
- Episode reward and final X-position are printed after each episode

---

## 📁 Project Structure

```
Super-Mario-Playing-RL-Agent/
│
├── mario.py              # Main script (environment, agent, training, testing)
├── best_mario.pth        # Best performing model checkpoint (saved during training)
├── final_mario.pth       # Final model checkpoint (saved at end of training)
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── venv310/              # Virtual environment (not tracked by git)
```

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | `0.00025` | Adam optimizer learning rate |
| `gamma` | `0.99` | Discount factor for future rewards |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `1e-5` | Epsilon decay per step |
| `tau` | `0.005` | Soft target update rate |
| `batch_size` | `32` | Mini-batch size for learning |
| `replay_buffer` | `100,000` | Maximum transitions stored |
| `frame_skip` | `4` | Frames to repeat each action |
| `frame_stack` | `4` | Number of frames stacked as input |
| `input_shape` | `(4, 84, 84)` | Stacked grayscale frame dimensions |

---

## 📊 Current Results

> ⚠️ **Note:** The agent was trained for only **1,000 episodes**, which is relatively short for a complex game like Super Mario Bros. As a result, the agent's performance is not optimal yet — it can navigate short distances but may struggle with later obstacles.

For significantly better performance, training for **5,000–10,000+ episodes** is recommended on a GPU.

**Saved checkpoints included:**
- `best_mario.pth` — Best model checkpoint from 1,000-episode training run
- `final_mario.pth` — Final model state at end of training

---

## 🚀 Future Improvements

- [ ] Train for more episodes (5,000+) on a GPU for better performance
- [ ] Implement **Prioritized Experience Replay (PER)** for smarter sampling
- [ ] Add **Dueling DQN** architecture for improved value estimation
- [ ] Experiment with **reward shaping** to encourage faster level completion
- [ ] Add **TensorBoard** logging for real-time training visualization
- [ ] Support **multiple levels** (World 1-1, 1-2, etc.)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ using PyTorch & Gymnasium | Super Mario Bros RL Agent
</p>
