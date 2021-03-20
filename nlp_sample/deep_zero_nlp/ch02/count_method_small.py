import numpy as np

from util import create_co_matrix, ppmi, preprocess


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, _ = preprocess(text)

    co_matrix = create_co_matrix(corpus, len(word_to_id))
    W = ppmi(co_matrix)
    U, S, V = np.linalg.svd(W)

    print('covariance matrix:')
    print(co_matrix)
    print('-'*30)
    print('PPMI:')
    print(W)
    print('-'*30)
    print('U:')
    print(U)
    print('DONE')
