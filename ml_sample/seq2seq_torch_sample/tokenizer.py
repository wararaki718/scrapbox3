from typing import List

import spacy


class Tokenizer:
    def __init__(self, lang: str, rev: bool = False):
        self._spacy = spacy.blank(lang)
        self._rev = rev
    
    def __call__(self, text: str) -> List[str]:
        result = [tok.text for tok in self._spacy.tokenizer(text)]
        if self._rev:
            return result[::-1]
        else:
            return result
