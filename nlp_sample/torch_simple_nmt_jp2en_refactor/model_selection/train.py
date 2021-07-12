import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Adam, criterion: nn.Module, clip: float) -> float:
    model.train()

    epoch_loss = 0.0
    for _, (src, trg) in enumerate(loader):
        src = src.cuda()
        trg = trg.cuda()

        optimizer.zero_grad()

        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])

        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

