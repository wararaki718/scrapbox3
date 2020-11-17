from typing import List

import MeCab

from .morph import Morph


class MecabAnalyzer:
    def __init__(self):
        self._mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
        _ = self._mecab.parse('') # avoid bug

    def __call__(self, context: str):
        return self.analyze(context)
    
    def analyze(self, context: str):
        context = self._preprocess(context)
        tokens = self._tokenize(context)
        return self._postprocess(tokens)
    
    def _tokenize(self, context: str) -> List[Morph]:
        result =  self._mecab.parse(context)
        tokens = map(lambda token: token.replace('\t', ',').split(','), result.split('\n')[:-2])
        morphs = map(lambda token: Morph(
            surface=token[0],
            part_of_speech=token[1],
            part_of_speech1=token[2],
            part_of_speech2=token[3],
            part_of_speech3=token[4],
            inflected_type=token[5],
            inflected_form=token[6],
            base_form=token[7],
            reading=token[8],
            phonetic=token[9]), tokens)
        return list(morphs)

    def _preprocess(self, context: str) -> str:
        return context

    def _postprocess(self, tokens: List[Morph]) -> List[Morph]:
        return tokens
