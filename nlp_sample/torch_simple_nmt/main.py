import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from download import download
from loader import get_dataloader
from model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 10
CLIP = 1


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Adam, criterion: nn.Module, clip: float) -> float:
    model.train()

    epoch_loss = 0.0
    for _, (src, trg) in enumerate(loader):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for _, (src, trg) in enumerate(loader):
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
        return epoch_loss / len(loader)


def epoch_time(start_time: int, end_time: int) -> Tuple[int, int]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time%60)
    return elapsed_mins, elapsed_secs


def main():
    train_data, val_data, test_data, de_vocab, en_vocab = download()
    train_loader = get_dataloader(train_data, de_vocab, en_vocab)
    val_loader = get_dataloader(val_data, de_vocab, en_vocab)
    test_loader = get_dataloader(test_data, de_vocab, en_vocab)

    model = get_model(len(de_vocab), len(en_vocab), DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropy(ignore_index=en_vocab.stoi["<pad>"])

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_loader, criterion)
        best_valid_loss = min(best_valid_loss, valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\tValid loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}")
    
    print(f"Best Valid loss: {best_valid_loss:.3f} | Valid PPL: {math.exp(best_valid_loss):7.3f}")

    test_loss = evaluate(model, test_loader, criterion)
    print(f"| Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
    print("DONE")


if __name__ == "__main__":
    main()
