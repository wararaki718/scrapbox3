import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from decorder import Decoder
from encoder import Encoder
from model import Seq2Seq
from model_selection import train, evaluate
from tokenizer import Tokenizer
from utils import count_parameters, epoch_time, init_weights


SEED = 42
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    g_tokenizer = Tokenizer("de", rev=True)
    e_tokenizer = Tokenizer("en")

    SRC = Field(
        tokenize=g_tokenizer,
        init_token='<sos>',
        eos_token='<eps>',
        lower=True
    )

    TRG = Field(
        tokenize=e_tokenizer,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'),
        fields=(SRC, TRG)
    )

    print(f'train: {len(train_data.examples)}')
    print(f'valid: {len(valid_data.examples)}')
    print(f'test : {len(test_data.examples)}')

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    print(f'SRC vocab: {len(SRC.vocab)}')
    print(f'TRG vocab: {len(TRG.vocab)}')

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
        device=torch.device('cuda')
    )

    encoder = Encoder(len(SRC.vocab), 256, 512, 2, 0.5)
    decoder = Decoder(len(TRG.vocab), 256, 512, 2, 0.5)

    model = Seq2Seq(encoder, decoder, torch.device('cuda')).cuda()
    model.apply(init_weights)

    print(count_parameters(model))

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    n_epochs = 10
    clip = 1

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'epoch {epoch}| time: {epoch_mins}m {epoch_secs}s')
        print(f'train loss: {train_loss} | train ppl: {math.exp(train_loss)}')
        print(f'valid loss: {valid_loss} | valid ppl: {math.exp(valid_loss)}')
        print()

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss = evaluate(model, test_iter, criterion)

    print(f'test loss: {test_loss} | test ppl: {math.exp(test_loss)}')
    print()
    print('DONE')


if __name__ == '__main__':
    main()
