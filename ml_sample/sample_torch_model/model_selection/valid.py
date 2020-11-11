import torch


def valid(model, X, y):
    model.eval()
    activation = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        y_pred = model(X)
        test_loss = activation(y_pred, y)
        print(f'[valid] average loss: {test_loss.item()}')
