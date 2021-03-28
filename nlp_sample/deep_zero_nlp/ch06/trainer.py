import time
from typing import Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from util import clip_grads


def remove_duplicate(params: List[np.ndarray], grads: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    params, grads = params[:], grads[:]

    while True:
        L = len(params)
        find_flg = False

        for i in range(0, L-1):
            for j in range(i+1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    params.pop(j)
                    grads.pop(j)
                    find_flg = True
                elif params[i].ndim == 2 and \
                     params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and \
                     np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    params.pop(j)
                    grads.pop(j)
                    find_flg = True
                
                if find_flg:
                    break
            if find_flg:
                break
        if not find_flg:
            break
    
    return params, grads


class RnnlmTrainer:
    def __init__(self, model: Any, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x: np.ndarray, t: np.ndarray, batch_size: int, time_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_x = np.empty((batch_size, time_size), dtype=int)
        batch_t = np.empty((batch_size, time_size), dtype=int)

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for tm in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, tm] = x[(offset+self.time_idx)%data_size]
                batch_t[i, tm] = t[(offset+self.time_idx)%data_size]
            self.time_idx += 1
        return batch_x, batch_t
    
    def fit(self, xs: np.ndarray, ts: np.ndarray, max_epoch: int=10, batch_size: int=20, time_size: int=35, max_grad: Optional[float]=None, eval_interval: int=20) -> None:
        data_size = len(xs)
        max_iters = data_size // (batch_size*time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0.0
        loss_count = 0

        start_time = time.time()
        for epoch in range(1, max_epoch+1):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters%eval_interval)==0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(f'| epoch {epoch} | iter {iters+1} / {max_iters} | time {elapsed_time}[s] | ppl {ppl}')
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
        self.current_epoch += 1

    def plot(self, ylim: Optional[np.ndarray] = None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel(f'iterations (x{self.eval_interval})')
        plt.ylabel('loss')
        plt.show()
