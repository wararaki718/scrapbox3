import numpy as np
from sklearn.datasets import load_wine
from sklearn.gaussian_process import GaussianProcessRegressor


def main():
    wine = load_wine()
    X = wine.data
    y = wine.target

    gpr = GaussianProcessRegressor()
    gpr.fit(X, y)
    for i in np.random.choice(X.shape[0], 10):
        x = X[i, :]
        print(f'real:{y[i]}, predict: {gpr.predict([x])}')
    print('DONE')


if __name__ == '__main__':
    main()
