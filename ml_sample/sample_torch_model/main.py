from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch

from model import NNModel
from model_selection import train, valid


def transform(x):
    x = torch.Tensor(x)
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def main():
    if torch.cuda.is_available():
        print('use cuda')

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    model = NNModel(4, 3).cuda()
    model = train(model, transform(X_train), transform(y_train).long())
    valid(model, transform(X_test), transform(y_test).long())
    print('DONE')


if __name__ == '__main__':
    main()
