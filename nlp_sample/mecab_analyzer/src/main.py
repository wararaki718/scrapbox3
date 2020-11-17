from analyzer import MecabAnalyzer


def main():
    # mecab with neologd
    analyzer = MecabAnalyzer()

    text = '私は、渋谷ストリームでランチを食べる。'
    tokens = analyzer(text)
    print(tokens)
    print('DONE')


if __name__ == '__main__':
    main()
