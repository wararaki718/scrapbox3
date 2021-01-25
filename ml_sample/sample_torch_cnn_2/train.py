import torch


def train(model: 'NNModel',
          loader: torch.utils.data.DataLoader,
          epochs: int) -> 'NNModel':
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for i, data in enumerate(loader):
            X, y = data

            optimizer.zero_grad()

            y_pred = model(X.cuda())
            loss = criterion(y_pred, y.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/2000}')
                running_loss = 0.0

    return model
