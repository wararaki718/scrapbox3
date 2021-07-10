import torch
from torch._C import device
import torch.nn as nn

from .attention import Attention
from .decoder import Decoder
from .encoder import Encoder
from .seq2seq import Seq2Seq


ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def get_model(input_dim: int, output_dim: int) -> Seq2Seq:
    encoder = Encoder(input_dim, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    decoder = Decoder(output_dim, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attention)

    model = Seq2Seq(encoder, decoder).cuda()
    model.apply(init_weights)
    
    return model
