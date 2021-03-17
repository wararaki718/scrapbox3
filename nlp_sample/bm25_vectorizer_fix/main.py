from bm25 import BM25Vectorizer


def main():
    vectorizer = BM25Vectorizer()
    sentences = [
        'Do you quarrel sir?',
        'Quarrel sir! no, sir!',
        'If you do, sir, I am for you: I serve as good a man as you.',
        'No better',
        'Well sir.'
    ]
    sentences = [sentence.lower() for sentence in sentences]
    vectorizer.fit(sentences)
    print(vectorizer.get_feature_names())

    x = ['quarrel sir']
    y = vectorizer.transform(x)
    print(y)

    print('DONE')


if __name__ == '__main__':
    main()
