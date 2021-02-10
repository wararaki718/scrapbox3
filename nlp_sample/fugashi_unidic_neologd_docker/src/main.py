from fugashi import Tagger


def main():
    tagger = Tagger()
    neologd_tagger = Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-unidic-neologd')

    text = '私は、渋谷ストリームでランチを食べる。'
    print('unidic:')
    print(tagger.parse(text))
    print()

    print('unidic-neologd:')
    print(neologd_tagger.parse(text))
    print('DONE')


if __name__ == '__main__':
    main()
