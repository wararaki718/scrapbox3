import re
import unicodedata
from typing import List, Tuple

import torch

from lang import Lang, EOS_TOKEN


def _unicode2ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c)
    )


def normalize(s: str) -> str:
    s = s.lower()
    s = s.strip()
    s = _unicode2ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def _sentence2indices(lang: Lang, sentence: str) -> List[int]:
    return [lang.word2index[word] for word in sentence.split(" ")]


def sentence2tensor(lang: Lang, sentence: str) -> torch.Tensor:
    indices = _sentence2indices(lang, sentence)
    indices.append(EOS_TOKEN)
    return torch.tensor(indices).long().view(-1, 1)


def pair2tensor(input_lang: Lang, output_lang: Lang, pair: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensor = sentence2tensor(input_lang, pair[0])
    output_tensor = sentence2tensor(output_lang, pair[1])
    return (input_tensor, output_tensor)
