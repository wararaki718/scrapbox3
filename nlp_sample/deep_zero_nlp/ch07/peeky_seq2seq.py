from typing import List

import numpy as np

from seq2seq import Encoder, Seq2seq
from time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss


class PeekyDecoder:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx = (np.random.randn(wordvec_size+hidden_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b = np.zeros(4*hidden_size).astype(float)
        affine_W = (np.random.randn(hidden_size+hidden_size, vocab_size) / np.sqrt(hidden_size)).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = []
        self.grads = []

        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs: np.ndarray, h: np.ndarray) -> np.ndarray:
        N, T = xs.shape
        N, H = h.shape
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore: np.ndarray) -> np.ndarray:
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)
        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h: np.ndarray, start_id: int, sample_size: int) -> List[int]:
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        
        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = PeekyDecoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
