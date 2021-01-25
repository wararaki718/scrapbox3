import torch


class NNModel(torch.nn.Module):
    def __init__(self) -> None:
        super(NNModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> float:
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)

        x = x.view(-1, self._num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _num_flat_features(self, x: torch.Tensor) -> int:
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    model = NNModel()
    print(model)
    print(list(model.parameters())[0].size())
