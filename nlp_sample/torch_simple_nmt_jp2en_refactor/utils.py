from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab


def epoch_time(start_time: int, end_time: int) -> Tuple[int, int]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time%60)
    return elapsed_mins, elapsed_secs


def show(batch_indices: torch.Tensor, vocab: Vocab):
    batch_indices = batch_indices.cpu()
    for indices in batch_indices:
        #print(indices)
        #print(indices.shape)
        print([vocab.itos[i] for i in indices])
        break # debug
    print()


def translate(model: nn.Module,
              loader: DataLoader,
              src_vocab: Vocab,
              trg_vocab: Vocab):
    with torch.no_grad():
        for _, (src, trg) in enumerate(loader):
            src = src.cuda()
            trg = trg.cuda()

            output = model(src, trg, 0)
            #print(src.shape)
            #print(trg.shape)
            #print(output.shape)

            output = output.topk(1, dim=2).indices.squeeze()
            #print(output.shape)

            print("source: ", end="")
            show(src, src_vocab)

            print("target: ", end="")
            show(trg, trg_vocab)

            print("output: ", end="")
            show(output, trg_vocab)
            break # debug
