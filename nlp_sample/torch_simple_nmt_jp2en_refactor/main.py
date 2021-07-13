import math
import time

import torch.nn as nn
import torch.optim as optim

from loader import get_dataloader
from model import get_model
from model_selection import evaluate, train
from preprocess import preprocessing
from utils import epoch_time, translate, sentence_translate


N_EPOCHS = 1
CLIP = 1


def main():
    train_data, val_data, test_data, ja_vocab, en_vocab, ja_tokenizer, en_tokenizer = preprocessing()
    train_loader = get_dataloader(train_data, ja_vocab, en_vocab)
    val_loader = get_dataloader(val_data, ja_vocab, en_vocab)
    test_loader = get_dataloader(test_data, ja_vocab, en_vocab)

    print(f"the number of ja_vocab: {len(ja_vocab.stoi)}")
    print(f"the number of en_vocab: {len(en_vocab.stoi)}")

    model = get_model(len(ja_vocab), len(en_vocab))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.stoi["<pad>"])

    best_valid_loss = float("inf")

    print("start training:")
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
    print()
    print("sample output:")
    translate(model, test_loader, ja_vocab, en_vocab)
    print()
    print("sentence translation:")
    sentence = "法嗣という師匠から弟子へと悟りの伝達が続き現在に至る"
    sentence_translate(model, sentence, ja_vocab, en_vocab, ja_tokenizer)
    print("DONE")


if __name__ == "__main__":
    main()
