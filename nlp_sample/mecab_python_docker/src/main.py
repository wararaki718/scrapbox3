import MeCab


def main():
    # mecab with neologd
    mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    mecab.parse('') # avoid bug

    text = '私は、渋谷ストリームでランチを食べる。'
    print(mecab.parse(text))
    print('DONE')


if __name__ == '__main__':
    main()
