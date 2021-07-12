import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for _, (src, trg) in enumerate(loader):
            src = src.cuda()
            trg = trg.cuda()

            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
        return epoch_loss / len(loader)
