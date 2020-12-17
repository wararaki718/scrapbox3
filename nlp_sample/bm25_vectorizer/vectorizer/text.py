from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25Vectorizer:
    def __init__(self, k1: float=1.2, b: float=0.75):
        self._k1 = k1
        self._b = b
        self._vectorizer = TfidfVectorizer()

    def fit(self, X: List[str]):
        self._avdl = self._vectorizer.fit_transform(X).sum(1).mean()

    def transform(self, q: str, X: List[str]):
        X_ = self._vectorizer.transform(X)
        D = X_.sum(1).A1
        q_ = self._vectorizer.transform([q])
        
        X_ = X_.tocsr()[:, q_.indices]
        denom = X_ + (self._k1 * (1-self._b + self._b*D/self._avdl))[:, None]
        idf = self._vectorizer._tfidf.idf_[None, q_.indices] - 1
        numer = X_.multiply(np.broadcast_to(idf, X_.shape)) * (self._k1 + 1)
        return (numer/denom).sum(1).A1
