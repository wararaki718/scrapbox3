import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int, batch_size: int=32, num_layers: int=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, wordvec_size)
        self.rnn = nn.RNN(wordvec_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = self.linear(x)
        return x

    def init_hidden(self, batch_size: int=None):
        if not batch_size:
            batch_size = self.batch_size
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.cuda()
