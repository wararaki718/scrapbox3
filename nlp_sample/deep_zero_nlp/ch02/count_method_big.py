from sklearn.utils.extmath import randomized_svd

from dataset import ptb
from util import most_similar, create_co_matrix, ppmi


def main():
    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    print('counting co-occurence...')
    co_matrix = create_co_matrix(corpus, vocab_size, window_size)
    print('calculating PPMI...')
    W = ppmi(co_matrix, verbose=True)

    print('calculating SVD ...')
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=42)

    word_vecs = U[:, :wordvec_size]

    queries = ['you', 'year', 'car', 'toyota']
    for query in queries:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    print('DONE')


if __name__ == '__main__':
    main()
