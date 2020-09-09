import numpy as np
from sklearn.naive_bayes import MultinomialNB


def main():
    # [we, meeting, click]
    X = np.array([[1, 1, 0], [1, 0, 1]])

    # 1: not spam, 2: spam
    y = np.array([1, 2])

    clf = MultinomialNB()
    clf.fit(X, y)

    z = np.array([[0, 0, 1]])
    print(clf.predict(z))
    print(clf.predict_proba(z))

    clf = MultinomialNB()
    clf.set_params(alpha=10.0)
    clf.fit(X, y)
    
    print(clf.predict(z))
    print(clf.predict_proba(z))


if __name__ == '__main__':
    main()
