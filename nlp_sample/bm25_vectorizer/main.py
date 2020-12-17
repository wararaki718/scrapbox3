from vectorizer import BM25Vectorizer


def main():
    texts = [
        "This is a pen",
        "That is an apple",
        "This is a good pencil",
        "The pen is good",
        "The pencil is bad",
        "The apple is blue",
        "This is a red apple"
    ]

    bm25_vectorizer = BM25Vectorizer()
    bm25_vectorizer.fit(texts[1:])
    
    print(bm25_vectorizer.transform(texts[0], texts))
    print('DONE')


if __name__ == '__main__':
    main()
