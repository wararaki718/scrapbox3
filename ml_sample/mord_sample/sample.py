from mord import LogisticAT
from sklearn.datasets import load_wine

# !pip install mord

def main():
    wine = load_wine()
    X = wine.data
    y = wine.target

    print(X)
    print(y)

    model = LogisticAT()
    model.fit(X, y)
    print(model.predict(X))


if __name__ == '__main__':
    main()
