from typing import Dict, List, Optional, Tuple

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


def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float=1e-8) -> np.ndarray:
    nx = x / (np.sqrt(np.sum(x**2))+eps)
    ny = y / (np.sqrt(np.sum(y**2))+eps)

    return np.dot(nx, ny)


def most_similar(query: str, word_to_id: Dict[str, int], id_to_word: Dict[int, str], word_matrix: np.ndarray, top: int=5) -> None:
    if query not in word_to_id:
        print(f'{query} is not found')
        return

    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'  {id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return

def analogy(a: str, b: str, c: str, word_to_id: Dict[str, int], id_to_word: Dict[int, str], word_matrix: np.ndarray, top: int=5, answer: Optional[str]=None) -> None:
    for word in (a, b, c):
        if word not in word_to_id:
            print(f"{word} is not found")
            return
    
    print(f"[analogy] {a}:{b} = {c}:?")
    a_vec = word_matrix[word_to_id[a]]
    b_vec = word_matrix[word_to_id[b]]
    c_vec = word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print(f'==>{answer}:{str(np.dot(word_matrix[word_to_id[answer]], query_vec))}')
    
    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        
        print(f' {id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return
