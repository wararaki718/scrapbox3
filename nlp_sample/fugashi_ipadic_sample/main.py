import inspect

from fugashi import GenericTagger
import ipadic


def main():
    tagger = GenericTagger(ipadic.MECAB_ARGS+' -Owakati')

    text = '私はご飯を食べます。'
    for word in tagger(text):
        print(word)
        # print(inspect.getmembers(word))
    print('DONE')


if __name__ == '__main__':
    main()
