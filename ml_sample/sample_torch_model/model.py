import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=5):
        super(NNModel, self).__init__()
        self.linear_1 = nn.Linear(n_input, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> float:
        x = self.linear_1(x)
        x = self.linear_2(x)
        return self.softmax(x)
