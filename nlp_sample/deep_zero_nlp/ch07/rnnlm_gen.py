from typing import List, Optional, Tuple

import numpy as np

from functions import softmax
from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    def generate(self, start_id: int, skip_ids: Optional[List[int]]=None, sample_size: int=100) -> List[int]:
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        
        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id: int, skip_ids: Optional[List[int]]=None, sample_size: int=100) -> List[int]:
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        
        return word_ids

    def get_state(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, state: Tuple[np.ndarray]) -> None:
        states = []
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)
