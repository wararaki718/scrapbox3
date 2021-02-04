import spacy


def main():
    text = '私はご飯を食べます。'
    nlp = spacy.blank('ja') # use sudachipy
    for word in nlp(text):
        print(word, word.lemma_, word.tag_, word.pos_)
    print()

    nlp = spacy.load('ja_ginza') # use ginza
    for word in nlp(text):
        print(word, word.lemma_, word.tag_, word.pos_)
    print('DONE')


if __name__ == '__main__':
    main()
