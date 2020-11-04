from sklearn.decomposition import IncrementalPCA, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def use_pca_with_function_transformer() -> Pipeline:
    def to_dense(X):
        return X.todense()

    pipe = Pipeline([
        ('cv', CountVectorizer()),
        ('ft', FunctionTransformer(to_dense)),
        ('pca', PCA(n_components=2))
    ])
    return pipe


def use_incremental_pca() -> Pipeline:
    pipe = Pipeline([
        ('cv', CountVectorizer()),
        ('ipca', IncrementalPCA(n_components=2, batch_size=4))
    ])
    return pipe


def use_truncated_svd() -> Pipeline:
    pipe = Pipeline([
        ('cv', CountVectorizer()),
        ('tsvd', TruncatedSVD(n_components=2))
    ])
    return pipe
