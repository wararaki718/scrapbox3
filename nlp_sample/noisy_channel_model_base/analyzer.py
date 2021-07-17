from typing import Generator

import spacy


class Analyzer:
    def __init__(self, model: str="ja_ginza"):
        self._tokenizer = spacy.load(model)
    
    def analyze(self, sentence: str) -> Generator[str, None, None]:
        for token in self._tokenizer(sentence):
            yield token.text
