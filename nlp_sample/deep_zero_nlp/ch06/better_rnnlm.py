import pickle

import numpy as np

from time_layers import TimeEmbedding, TimeDropout, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss


class BetterRnnlm:
    def __init__(self, vocab_size: int=10000, wordvec_size: int=100, hidden_size: int=100, dropout_ratio: float=1.0) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx1 = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh1 = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b1 = np.zeros(4*hidden_size).astype(float)
        lstm_Wx2 = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh2 = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b2 = np.zeros(4*hidden_size).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs: np.ndarray, train_flg: bool=False) -> np.ndarray:
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs: np.ndarray, ts: np.ndarray, train_flg: bool=True) -> float:
        score = self.predict(xs, train_flg)
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
