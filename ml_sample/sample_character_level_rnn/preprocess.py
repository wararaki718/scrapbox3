import torch

from util import ALL_LETTERS


def letter2index(letter: str) -> int:
    return ALL_LETTERS.find(letter)


def letter2tensor(letter: str) -> torch.Tensor:
    tensor = torch.zeros(1, len(ALL_LETTERS))
    tensor[0][letter2index(letter)] = 1
    return tensor


def line2tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, len(ALL_LETTERS))
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor
