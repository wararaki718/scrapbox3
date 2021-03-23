from collections import Counter
from typing import List

import numpy as np

from layers import Embedding, SigmoidWithLoss


class EmbeddingDot:
    def __init__(self, W: np.ndarray) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h: np.ndarray, idx: List[int]):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h, target_W = self.cache

        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus: List[int], power: float, sample_size: int) -> None:
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = Counter()
        for word_id in corpus:
            counts[word_id] += 1
        
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target: np.ndarray) -> np.ndarray:
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W: np.ndarray, corpus: List[int], power: float=0.75, sample_size: int=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h: np.ndarray, target: np.ndarray) -> np.ndarray:
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, 1]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)
        return loss

    def backward(self, dout: int=1)-> float:
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


if __name__ == '__main__':
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2

    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negative_sample = sampler.get_negative_sample(target)
    print(negative_sample)
