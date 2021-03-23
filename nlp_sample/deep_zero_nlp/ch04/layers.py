from typing import List
import numpy as np

from functions import cross_entropy_error


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


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1-self.y, self.y], self.t)
        return self.loss

    def backward(self, dout: int=1) -> np.ndarray:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx
