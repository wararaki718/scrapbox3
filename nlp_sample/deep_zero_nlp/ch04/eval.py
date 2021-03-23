import pickle

from util import analogy, most_similar


def main() -> None:
    with open('cbow_params.pkl', 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']
    
    queries = ['you', 'year', 'car', 'toyota']
    for query in queries:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
    analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
    analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
    analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)


if __name__ == '__main__':
    main()
