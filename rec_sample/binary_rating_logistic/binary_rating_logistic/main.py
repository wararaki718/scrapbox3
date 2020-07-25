import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv('data/data.csv')
    y = df['rating'].values
    X = df.drop('rating', axis=1).values

    lr = LogisticRegression()
    lr.fit(X, y)

    x_1 = np.array([[1, 0, 1, 0, 1, 0]])
    x_2 = np.array([[0, 0, 0, 0, 1, 0]])
    x_3 = np.array([[1, 1, 1, 0, 0, 0]])

    print(lr.predict_proba(x_1))
    print(lr.predict_proba(x_2))
    print(lr.predict_proba(x_3))
    print('DONE')


if __name__ == '__main__':
    main()



    