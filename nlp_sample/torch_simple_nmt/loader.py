from typing import Any, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab


BATCH_SIZE = 128
PAD_INDEX = None
BOS_INDEX = None
EOS_INDEX = None


def generate_batch(data_batch: List[Tuple[torch.Tensor]]) -> Tuple:
    global PAD_INDEX
    global BOS_INDEX
    global EOS_INDEX
    de_batch = []
    en_batch = []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_INDEX]), de_item, torch.tensor([EOS_INDEX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_INDEX]), en_item, torch.tensor([EOS_INDEX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_INDEX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_INDEX)


def get_dataloader(data: List[Tuple[torch.Tensor]], de_vocab: Vocab, en_vocab: Vocab) -> DataLoader:
    global PAD_INDEX
    global BOS_INDEX
    global EOS_INDEX
    PAD_INDEX = de_vocab["<pad>"]
    BOS_INDEX = de_vocab["<bos>"]
    EOS_INDEX = de_vocab["<eos>"]

    data_iter = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    return data_iter
