import torch
from torchtext.data import BucketIterator

from model import Seq2Seq


def train(model: Seq2Seq,
          iterator: BucketIterator,
          optimizer: torch.optim.Adam,
          criterion: torch.nn.CrossEntropyLoss,
          clip: int) -> float:
    model.train()
    epoch_loss = 0.0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model: Seq2Seq,
             iterator: BucketIterator,
             criterion: torch.nn.CrossEntropyLoss) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
