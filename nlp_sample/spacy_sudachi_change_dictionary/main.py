from spacy.lang.ja import Japanese


def show(nlp: Japanese, text: str):
    for word in nlp(text):
        print(word, word.lemma_, word.tag_, word.pos_)
    print()


def main():
    text = "私は機能性食品を購入した。"
    a_nlp = Japanese(meta={"tokenizer": {"config": {"split_mode": "A"}}})
    b_nlp = Japanese(meta={"tokenizer": {"config": {"split_mode": "B"}}})
    c_nlp = Japanese(meta={"tokenizer": {"config": {"split_mode": "C"}}})

    print('mode: A:')
    show(a_nlp, text)

    print('mode: B:')
    show(b_nlp, text)

    print('mode: C:')
    show(c_nlp, text)

    print('DONE')


if __name__ == '__main__':
    main()
