from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from loader import get_dataloader


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


def sentence_translate(model: nn.Module,
                       sentence: str,
                       source_vocab: Vocab,
                       target_vocab: Vocab,
                       tokenizer: Any):
    src = torch.tensor([source_vocab[token] for token in tokenizer(sentence)], dtype=torch.long)
    trg = torch.tensor([target_vocab["<bos>"]], dtype=torch.long)
    # eos_index = target_vocab["<eos>"]

    loader = get_dataloader([(src, trg)], source_vocab, target_vocab)

    with torch.no_grad():
        for _, (src, trg) in enumerate(loader):
            #print(src.shape)
            #print(trg.shape)
            output = model(src.cuda(), trg.cuda(), 0, max_length=128)
            #print(output.shape)

            output = output.topk(1, dim=2).indices.squeeze()
            output = torch.reshape(output, (1, -1))
            #print(output.shape)
            print(f"before: {sentence}")
            print("after: ", end="")
            show(output, target_vocab)


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
            show(torch.t(src), src_vocab)

            print("target: ", end="")
            show(torch.t(trg), trg_vocab)

            print("output: ", end="")
            show(torch.t(output), trg_vocab)
            break # debug
