from fugashi import Tagger


def main():
    tagger = Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-unidic-neologd')

    text = '私は、渋谷ストリームでランチを食べる。'
    print(tagger.parse(text))
    print('DONE')


if __name__ == '__main__':
    main()
