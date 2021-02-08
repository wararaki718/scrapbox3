import random

import torch
import torch.nn as nn
from torchtext.data import Field

from decorder import Decoder
from encoder import Encoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src: Field, trg: Field, teacher_forcing_ratio: float=0.5) -> torch.Tensor:
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input_ = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_ = trg[t] if teacher_force else top1
        
        return outputs
