from typing import List

import numpy as np

from functions import softmax


class Softmax:
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = softmax(x)
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


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
