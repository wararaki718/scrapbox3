import torch


def test(model: 'NNModel',
         loader: torch.utils.data.DataLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            X, y = data
            y_pred = model(X.cuda())
            _, y_label = torch.max(y_pred.data, 1)
            total += y_label.size(0)
            correct += (y_label == y.cuda()).sum().item()
    
    print(f'accuracy: {100*correct/total}%')
