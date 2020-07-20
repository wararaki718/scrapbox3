import numpy as np
from sklearn.datasets import load_breast_cancer


def main():
    X = load_breast_cancer().data
    print(f'X shape is {X.shape}')

    # SVD
    U, S, V_h = np.linalg.svd(X)

    # dim reduction
    dim = 15
    X_new = np.dot(X, V_h[:, :dim])

    print(f'X_new shape is {X_new.shape}')
    print('DONE')


if __name__ == '__main__':
    main()
