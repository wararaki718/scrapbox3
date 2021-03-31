from typing import List
import numpy as np


class Embedding:
    def __init__(self, W: np.ndarray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx: List[int]) -> np.ndarray:
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout: np.ndarray) -> None:
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        
        return None
