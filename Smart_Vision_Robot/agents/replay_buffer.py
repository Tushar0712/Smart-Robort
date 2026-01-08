# agents/replay_buffer.py
import numpy as np
import random
from collections import deque

class SequenceReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state_seq, action, reward, next_state_seq, done):
        """
        Store (state_seq, action, reward, next_state_seq, done)
        Make copies so outside modification doesn't change stored arrays.
        """
        self.buffer.append((np.array(state_seq, copy=True),
                            int(action),
                            float(reward),
                            np.array(next_state_seq, copy=True),
                            float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_seq, actions, rewards, next_state_seq, dones = map(lambda x: np.array(x), zip(*batch))
        return state_seq, actions, rewards, next_state_seq, dones

    def __len__(self):
        return len(self.buffer)
