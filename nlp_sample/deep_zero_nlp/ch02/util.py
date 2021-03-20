from typing import Dict, Tuple

import numpy as np


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


def create_co_matrix(corpus: np.ndarray, vocab_size: int, window_size: int=1) -> np.ndarray:
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


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


def ppmi(C: np.ndarray, verbose: bool = False, eps: float = 1e-8) -> np.ndarray:
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print(f'{100*cnt/total:.1f} done')
    return M


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    print(word_to_id)
    print(id_to_word)

    co_matrix = create_co_matrix(corpus, len(word_to_id))
    print(co_matrix)

    c0 = co_matrix[word_to_id['you']]
    c1 = co_matrix[word_to_id['i']]
    print(cos_similarity(c0, c1))

    most_similar('you', word_to_id, id_to_word, co_matrix, top=5)

    W = ppmi(co_matrix)
    print('covariance matrix:')
    print(co_matrix)
    print('-'*30)
    print('PPMI:')
    print(W)

    print('DONE')
