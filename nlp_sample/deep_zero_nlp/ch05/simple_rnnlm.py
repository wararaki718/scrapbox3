import numpy as np

from time_layers import TimeRNN, TimeAffine, TimeSoftmaxWithLoss, TimeEmbedding


class SimpleRnnlm:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        rnn_Wx = (np.random.randn(wordvec_size, hidden_size)/np.sqrt(wordvec_size)).astype(float)
        rnn_Wh = (np.random.randn(hidden_size, hidden_size)/np.sqrt(hidden_size)).astype(float)
        rnn_b = np.zeros(hidden_size).astype(float)
        affine_W = (np.random.randn(hidden_size, vocab_size)/np.sqrt(hidden_size)).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout:int=1) -> np.ndarray:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) -> None:
        self.rnn_layer.reset_state()
