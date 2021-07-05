from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.embedding(input_).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size)


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float=0.1, max_length: int=10):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = nn.Linear(hidden_size*2, max_length)
        self.attention_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attention_applied = torch.bmm(
            attention_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights
    
    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size)
