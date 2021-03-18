from typing import Any, List, Union
import warnings

import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency, CountVectorizer
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args, FLOAT_DTYPES



class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1: float=2.0, b: float=0.75, use_idf: bool=True):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X: Union[sps.csr_matrix, np.ndarray]) -> 'BM25Transformer':
        if not sps.issparse(X):
            X = sps.csr_matrix(X)
        
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sps.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X: Union[sps.csr_matrix, np.ndarray], copy: bool=True) -> sps.csr_matrix:
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sps.csr_matrix(X, copy=copy)
        else:
            X = sps.csr_matrix(X, dtype=np.float64, copy=copy)
        
        n_features = X.shape[1]

        dl = X.sum(axis=1)
        sz = X.indptr[1:] - X.indptr[0:-1]
        rep = np.repeat(np.asarray(dl), sz)

        avgdl = np.average(dl)
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sps.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')
            
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError(f"Input has n_features={n_features} while the model has been trained with n_features={expected_n_features}")
            X = X * self._idf_diag
        
        return X


class BM25Vectorizer(CountVectorizer):
    @_deprecate_positional_args
    def __init__(self, *, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, k1: float=2.0, b: float=0.75, use_idf=True):

        super().__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._bm25 = BM25Transformer(k1=k1, b=b, use_idf=use_idf)

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, raw_documents: List[str], y: Any=None):
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self

    def fit_transform(self, raw_documents: List[str], y: Any=None):
        self._check_params()
        self._warn_for_unused_params()
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self._bm25.transform(X, copy=False)

    def transform(self, raw_documents: List[str]):
        check_is_fitted(self, msg='The BM25 is not fitted')
        X = super().transform(raw_documents)
        return self._bm25.transform(X, copy=False)
