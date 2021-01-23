import torch

from nn_model import NNModel


def main():
    X = torch.randn(1, 1, 32, 32)
    y = torch.randn(10).view(1, -1)

    model = NNModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)

    print(y_pred)
    print(loss)

    loss.backward()
    optimizer.step()

    print('DONE')


if __name__ == '__main__':
    main()
