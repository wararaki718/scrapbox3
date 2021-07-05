from typing import List, Tuple

from lang import Lang
from util import normalize


def read_langs(lang1: str, lang2: str, reverse_: bool=False) -> Tuple[Lang, Lang, List[List[str]]]:
    print("Reading lines...")

    pairs = []
    filename = f"data/{lang1}-{lang2}.txt"
    with open(filename, "rt", encoding="utf-8") as f:
        for line in f:
            pairs.append([normalize(s) for s in line.split("\t")])
        
    if reverse_:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs
