from typing import List

import numpy as np

from attention_layer import TimeAttention
from seq2seq import Encoder, Seq2seq
from time_layers import TimeLSTM, TimeAffine, TimeSoftmaxWithLoss, TimeEmbedding


class AttentionEncoder(Encoder):
    def forward(self, xs: np.ndarray) -> np.ndarray:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs) -> np.ndarray:
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size) -> None:
        embed_W = (np.random.randn(vocab_size, wordvec_size)/100).astype(float)
        lstm_Wx = (np.random.randn(wordvec_size, 4*hidden_size)/np.sqrt(wordvec_size)).astype(float)
        lstm_Wh = (np.random.randn(hidden_size, 4*hidden_size)/np.sqrt(hidden_size)).astype(float)
        lstm_b = np.zeros(4*hidden_size).astype(float)
        affine_W = (np.random.randn(2*hidden_size, vocab_size)/np.sqrt(2*hidden_size)).astype(float)
        affine_b = np.zeros(vocab_size).astype(float)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]
        
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, xs: np.ndarray, enc_hs: np.ndarray) -> np.ndarray:
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore: np.ndarray) -> np.ndarray:
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        denc_hs[:, -1] += self.lstm.dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs: np.ndarray, start_id: int, sample_size: int) -> List[int]:
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        
        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int) -> None:
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
