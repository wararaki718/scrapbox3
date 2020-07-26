import numpy as np
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture


def main():
    wine = load_wine()
    X = wine.data
    y = wine.target

    gm = GaussianMixture(n_components=np.unique(y).shape[0])
    gm.fit(X, y)

    for i in np.random.choice(X.shape[0], 3):
        x = X[i, :]
        print(gm.predict_proba([x]))


if __name__ == '__main__':
    main()
