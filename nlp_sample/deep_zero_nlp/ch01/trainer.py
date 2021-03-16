import time
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from optimizer import SGD
from two_layer_net import TwoLayerNet
from utils import clip_grads


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


class Trainer:
    def __init__(self, model: TwoLayerNet, optimizer: SGD):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x: np.ndarray, t: np.ndarray, max_epoch: int=10, batch_size: int=32, max_grad: Optional[float]=None, eval_interval: int=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0.0
        loss_count = 0

        start_time = time.time()
        for epoch in range(1, max_epoch+1):
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters%eval_interval)==0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(f'| epoch {epoch} | iter {iters+1} / {max_iters} | time {elapsed_time}[s] | loss {avg_loss}')
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
        self.current_epoch += 1

    def plot(self, ylim: Optional[np.ndarray] = None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x{self.eval_interval})')
        plt.ylabel('loss')
        plt.show()
