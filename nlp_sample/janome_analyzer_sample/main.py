from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter
from janome.tokenfilter import CompoundNounFilter, LowerCaseFilter
from janome.tokenizer import Tokenizer


def main():
    char_filters = [UnicodeNormalizeCharFilter()]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), LowerCaseFilter()]
    analyzer = Analyzer(
        char_filters=char_filters,
        tokenizer=tokenizer,
        token_filters=token_filters
    )

    text = '私は、渋谷ストリームでランチを食べる。'
    for token in analyzer.analyze(text):
        print(token)
    print('DONE')


if __name__ == '__main__':
    main()
