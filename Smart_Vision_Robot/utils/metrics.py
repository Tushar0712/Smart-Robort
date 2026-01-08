# utils/metrics.py
import numpy as np
from collections import deque

class RunningStats:
    def __init__(self, window=100):
        self.window = window
        self.deque = deque(maxlen=window)

    def add(self, val):
        self.deque.append(val)

    def mean(self):
        if not self.deque:
            return 0.0
        return float(np.mean(self.deque))

    def last(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]
