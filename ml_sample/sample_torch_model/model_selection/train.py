import torch
import torch.optim as optim


def train(model, X, y, epochs: int=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters())

    model.train()
    train_loss = 0.0
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        print(f'[train] {epoch}-th epoch: {loss.item()}')
    print(f'[train] average loss: {train_loss/epochs}')
    
    return model
