from typing import List
import numpy as np


class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
