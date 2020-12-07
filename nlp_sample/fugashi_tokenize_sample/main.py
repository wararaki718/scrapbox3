import inspect

from fugashi import Tagger


def main():
    tagger = Tagger()
    wakati_tagger = Tagger('-Owakati')
    text = '私はご飯を食べます。'
    
    result = wakati_tagger.parse(text)
    print('result1(parse + wakati):')
    print(result)
    print(type(result))
    print()

    result = tagger.parse(text)
    print('result2(parse):')
    print(result)
    print(type(result))
    print()

    result = wakati_tagger(text)
    print('result3(_call_+wakati):')
    print(result)
    print(type(result))
    print(inspect.getmembers(result[0]))
    print(type(result[0]))
    print()

    result = tagger(text)
    print('result4(_call_):')
    print(result)
    print(type(result))
    print(inspect.getmembers(result[0]))
    print(type(result[0]))
    print()
    print('DONE')


if __name__ == '__main__':
    main()
