import numpy as np

from layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype(float)
        W_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype(float)

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [
            self.in_layer0,
            self.in_layer1,
            self.out_layer,
            self.loss_layer
        ]
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout: int=1) -> None:
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
