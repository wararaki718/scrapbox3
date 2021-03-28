import sys
from typing import Any, List

import numpy as np


def clip_grads(grads: List[np.ndarray], max_norm: float) -> None:
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate


def eval_perplexity(model: Any, corpus: np.ndarray, batch_size: int=10, time_size: int=35) -> np.ndarray:
    print('evaluating perplexity...')
    corpus_size = len(corpus)
    total_loss = 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=int)
        ts = np.zeros((batch_size, time_size), dtype=int)
        time_offset = iters * time_size
        offsets = [time_offset + (i*jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write(f"\r{iters} / {max_iters}")
        sys.stdout.flush()
    
    print('')
    ppl = np.exp(total_loss/max_iters)
    return ppl
