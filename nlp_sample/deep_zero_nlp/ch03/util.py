from typing import Dict, List, Tuple

import numpy as np


def clip_grads(grads: List[np.ndarray], max_norm: float):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text: str) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = dict()
    id_to_word = dict()
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_context_target(corpus: np.ndarray, window_size: int=1) -> Tuple[np.ndarray, np.ndarray]:
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus: np.ndarray, vocab_size: int) -> np.ndarray:
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print('preprocess:')
    print(corpus)
    print(word_to_id)
    print(id_to_word)

    contexts, target = create_context_target(corpus, window_size=1)
    print('create_context_target:')
    print(contexts)
    print(target)

    one_hot_target = convert_one_hot(target, len(word_to_id))
    one_hot_contexts = convert_one_hot(contexts, len(word_to_id))
    print('convert_one_hot:')
    print(one_hot_target)
    print(one_hot_contexts)

    print('DONE')
    