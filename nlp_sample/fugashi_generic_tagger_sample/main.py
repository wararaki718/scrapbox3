from fugashi import Tagger, GenericTagger


def main():
    text = 'softbank'
    tagger = Tagger()
    gtagger = GenericTagger()

    print('Tagger:')
    print(tagger.parse(text))
    for word in tagger(text):
        print(word.surface)
        print(word.feature)
    print()

    print('GenericTagger:')
    print(gtagger.parse(text))
    for word in gtagger(text):
        print(word.surface)
        print(word.feature)
    print()
    print('DONE')


if __name__ == '__main__':
    main()
