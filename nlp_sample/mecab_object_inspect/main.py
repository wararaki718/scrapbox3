import MeCab


def main():
    wakati_tagger = MeCab.Tagger('-Owakati')
    text = '私はご飯を食べます。'

    output1 = wakati_tagger.parse(text)
    print(type(output1))

    tagger = MeCab.Tagger()
    output2 = tagger.parse(text)
    print(type(output2))
    print('DONE')


if __name__ == '__main__':
    main()
