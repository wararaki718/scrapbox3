import torch


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.max_pool2d(torch.relu(self.conv1(x)), kernel_size=(2, 2))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
