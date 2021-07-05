import math
import random
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from lang import Lang, SOS_TOKEN, EOS_TOKEN
from model import AttentionDecoder, Encoder
from util import pair2tensor


def train(input_tensor: torch.Tensor,
          target_tensor: torch.Tensor,
          encoder: Encoder,
          decoder: AttentionDecoder,
          encoder_optimizer: optim.SGD,
          decoder_optimizer: optim.SGD,
          criterion: nn.NLLLoss,
          tearch_forcing_ratio: float=0.5,
          max_length: int=10):
    encoder_hidden = encoder.initHidden().cuda()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).cuda()
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[i],
            encoder_hidden
        )
        encoder_outputs[i] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_TOKEN]]).cuda()
    use_learning_rate = random.random() < tearch_forcing_ratio
    decoder_hidden = encoder_hidden

    loss = 0.0
    if use_learning_rate:
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == EOS_TOKEN:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iterators(pairs: List[List[str]],
                    encoder: Encoder,
                    decoder: AttentionDecoder,
                    lang_input: Lang,
                    lang_target: Lang,
                    n_iters: int,
                    print_every: int=10000,
                    learning_rate: float=0.5):
    start = time.time()
    loss_total = 0.0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [pair2tensor(lang_input, lang_target, random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for i in range(1, n_iters+1):
        training_pair = training_pairs[i-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor.cuda(),
            target_tensor.cuda(),
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion
        )
        loss_total += loss

        if i % print_every == 0:
            loss_average = loss_total / print_every
            loss_total = 0.0
            now = time.time() - start
            minute = math.floor(now/60)
            second = math.floor(now%60)
            print(f"({minute}m {second}s) ({i} {int(i/n_iters*100)}%) {loss_average}")
