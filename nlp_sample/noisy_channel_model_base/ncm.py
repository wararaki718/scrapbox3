import itertools
from typing import Dict, List, Optional

from nltk.lm import Vocabulary
from nltk.lm.models import MLE
from nltk.util import ngrams

from .analyzer import Analyzer


# Noisy Channel Model
class NCM:
    def __init__(self,
                 analyzer: Optional[Analyzer]=None,
                 n: int=3,
                 tp: int=0,
                 ts: int=-2,
                 n_candidate_char: int=300,
                 n_candidate_word: int=5):
        self._n_gram = n
        self._tp = tp
        self._ts = ts
        self._n_candidate_char = n_candidate_char
        self._n_candidate_word = n_candidate_word

        if analyzer is None:
            self._analyzer = Analyzer()
        else:
            self._analyzer = analyzer

    def fit(self, sentences: List[str]):
        char_vocabulary = Vocabulary(itertools.chain.from_iterable(sentences))
        char_ngram = [ngrams(sentence, self._n_gram) for sentence in sentences]
        
        self._lm = MLE(order=self._n_gram, vocabulary=char_vocabulary)
        self._lm.fit(char_ngram)

        tokenized_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue

            tokenized_sentence = [token for token in self._analyzer.analyze(sentence)]
            tokenized_sentence = ["__BOS__"] + tokenized_sentence + ["__EOS__"]
            tokenized_sentences.append(tokenized_sentence)
        
        word_vocabulary = Vocabulary(itertools.chain.from_iterable(tokenized_sentences))
        word_ngram = [ngrams(sentence, self._n_gram) for sentence in tokenized_sentences]

        self._wlm = MLE(order=self._n_gram, vocabulary=word_vocabulary)
        self._wlm.fit(word_ngram)


    def _detect(self, sentence: str) -> List[int]:
        scores = [0 for _ in range(len(sentence))]

        for i, context in enumerate(ngrams(sentence, self._n_gram)):
            p = self._lm.score(context[-1], context[:-1])
            if p < self._tp:
                for j in range(self._n_gram):
                    scores[i+j] -= 1
        return scores

    def _calc_Pf(self, cm: List[str]) -> float:
        pf = 1.0
        for i in range(2, len(cm)):
            pf *= self._lm.score(cm[i], (cm[i-2], cm[i-1]))
        return pf
    
    def _calc_Pb(self, cm: List[str]) -> float:
        pb = 1.0
        for i in range(2, len(cm)):
            pb *= self._lm.score(cm[i-2], (cm[i-1], cm[i]))
        return pb

    def generate(self, sentence: str, scores: List[int]) -> Dict[int, List[str]]:
        candidates = dict() # {pos: [word]}
        
        for i in range(2, len(scores)-2):
            if scores[i] > self._ts:
                continue

            m1s = []
            context = (sentence[i-2], sentence[i-1])
            for m in self._lm.context_counts(self._lm.vocab.lookup(context)):
                p = self._lm.score(m, context)
                if p <= 0:
                    continue
                m1s.append([m, p])
            m1s = sorted(m1s, key=lambda x: x[1])[::-1][:self._n_candidate_char]

            m2s = []
            for m, p in m1s:
                context = (sentence[i-1], m)
                for m2 in self._lm.context_counts(self._lm.vocab.lookup(context)):
                    p2 = self._lm.score(m2, context)
                    if p2 <= 0:
                        continue
                    m2s.append([m, m2, p*p2])
            m2s = sorted(m2s, key=lambda x: x[2])[::-1][:self._n_candidate_char]

            pfs = []
            pbs = []
            for m, _ in m1s:
                pf = self._calc_Pf([sentence[i-2], sentence[i-1], m, sentence[i+1], sentence[i+2]])
                pb = self._calc_Pb([sentence[i-2], sentence[i-1], m, sentence[i+1], sentence[i+2]])
                pfs.append([m, pf])
                pbs.append([m, pb])
            pfs = sorted(pfs, key=lambda x: x[1])[::-1][:self._n_candidate_word]
            pbs = sorted(pbs, key=lambda x: x[1])[::-1][:self._n_candidate_word]
            candidates[i] = list(set(m for m, _ in pfs+pbs))

            pfs = []
            pbs = []
            for m, m2, _ in m2s:
                pf = self._calc_Pf([sentence[i-2], sentence[i-1], m, m2, sentence[i+2], sentence[i+3]])
                pb = self._calc_Pb([sentence[i-2], sentence[i-1], m, m2, sentence[i+2], sentence[i+3]])
                pfs.append([m+m2, pf])
                pbs.append([m+m2, pb])
            pfs = sorted(pfs, key=lambda x: x[1])[::-1][:self._n_candidate_word]
            pbs = sorted(pbs, key=lambda x: x[1])[::-1][:self._n_candidate_word]
            candidates[i].extend(list(set(w for w, _ in pfs+pbs)))

        return candidates

    def select(self,
               sentence: str,
               candidates: Dict[int, List[str]]) -> List[List[str, float]]:
        results = []
        for i, words in candidates.items():
            for word in words:
                l = sentence[:i] + word + sentence[i+len(word):]
                tokens = [token for token in self._analyzer.analyze(l)]

                pl = 1.0
                is_calc = False
                for context in ngrams(tokens, self._n_gram):
                    pl *= self._wlm.score(context[-1], context[:-1])
                    is_calc = True
                
                if is_calc and pl > 0:
                    results.append([l, pl])
        return results
