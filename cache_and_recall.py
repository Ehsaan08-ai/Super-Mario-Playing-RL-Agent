from tensordict import TensorDict
from torch.rl import TensorDictReplayBuffer, LazyMemmapStorage
from Agent import Mario


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(200000, device="cpu"))
        self.batch_size = 32
