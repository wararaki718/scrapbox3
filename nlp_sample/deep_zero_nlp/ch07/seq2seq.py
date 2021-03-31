from typing import List

import numpy as np

from base_model import BaseModel
from time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss


class Encoder:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b = np.zeros(4*hidden_size).astype(float)

        self.embed = TimeEmbedding(embed_W).astype(float)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None
    
    def forward(self, xs: np.ndarray) -> np.ndarray:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh: np.ndarray) -> None:
        dhs = np.zero_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout) # return None
        return dout


class Decoder:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b = np.zeros(4*hidden_size).astype(float)
        affine_W = (np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = []
        self.grads = []

        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, h: np.ndarray) -> np.ndarray:
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore: np.ndarray) -> np.ndarray:
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h: np.ndarray, start_id: int, sample_size: int) -> List[int]:
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
        
        return sampled


class Seq2seq(BaseModel):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> np.ndarray:
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout: float=1) -> np.ndarray:
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs: np.ndarray, start_id: int, sample_size: int) -> List[int]:
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
