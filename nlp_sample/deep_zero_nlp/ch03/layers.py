import numpy as np

from functions import cross_entropy_error, softmax


class MatMul:
    def __init__(self, W: np.ndarray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, = self.params
        out = np.dot(x, W)
        self.x = x

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None
    
    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout:int=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
