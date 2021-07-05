from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size)
