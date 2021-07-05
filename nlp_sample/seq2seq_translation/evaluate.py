import random
from typing import List, Tuple

import torch

from lang import Lang, SOS_TOKEN, EOS_TOKEN
from model import AttentionDecoder, Encoder
from util import sentence2tensor


def evaluate(encoder: Encoder,
             decoder: AttentionDecoder,
             input_lang: Lang,
             output_lang: Lang,
             sentence: str,
             max_length: int=10) -> Tuple[List[str], torch.Tensor]:
    input_tensor = sentence2tensor(input_lang, sentence).cuda()
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden().cuda()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).cuda()

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[i],
            encoder_hidden
        )
        encoder_outputs[i] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]]).cuda()
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for i in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[i] = decoder_attention.data
        topv, topi = decoder_output.topk(1)
        if topi.item() == EOS_TOKEN:
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(output_lang.index2word[topi.item()])
        
        decoder_input = topi.squeeze().detach()
    
    return decoded_words, decoder_attentions[:max_length]


def evaluate_randomly(encoder: Encoder,
                      decoder: AttentionDecoder,
                      input_lang: Lang,
                      output_lang: Lang,
                      pairs: List[List[str]],
                      n: int=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
        output_sentence = " ".join(output_words)
        print('<', output_sentence)
        print()
