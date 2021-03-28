import pickle

import numpy as np

from time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss


class Rnnlm:
    def __init__(self, vocab_size: int=10000, wordvec_size: int=100, hidden_size: int=100) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b = np.zeros(4*hidden_size).astype(float)
        affine_W = (np.random.randn(hidden_size, vocab_size)/np.sqrt(hidden_size)).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout: float=1.0) -> float:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) -> None:
        self.lstm_layer.reset_state()

    def save_params(self, filename: str='Rnnlm.pkl') -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)


    def load_params(self, filename: str='Rnnlm.pkl') -> None:
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
