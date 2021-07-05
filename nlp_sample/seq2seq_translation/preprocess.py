from typing import List, Tuple

from lang import Lang
from read import read_langs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(pair: List[str]) -> bool:
    return len(pair[0].split(" ")) < MAX_LENGTH and \
           len(pair[1].split(" ")) < MAX_LENGTH and \
           pair[0].startswith(eng_prefixes)


def filter_pairs(pairs: List[List[str]]) -> List[List[str]]:
    return [pair for pair in pairs if filter_pair(pair)]


def preprocessing(lang1: str, lang2: str, reverse_: bool=False) -> Tuple[Lang, Lang, List[List[str]]]:
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse_)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filter_pairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")

    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs
