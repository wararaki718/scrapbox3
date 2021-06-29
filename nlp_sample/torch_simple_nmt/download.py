import io
import urllib.parse
from collections import Counter
from functools import partial
from typing import Any, List, Tuple, Union

import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive


URL_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
TRAIN_URLS = [
    "train.de.gz",
    "train.en.gz"
]
VAL_URLS = [
    "val.de.gz",
    "val.en.gz"
]
TEST_URLS = [
    "test_2016_flickr.de.gz",
    "test_2016_flickr.en.gz"
]


def build_vocab(filepath: str, tokenizer: Any) -> Vocab:
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for line in f:
            counter.update(tokenizer(line))
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process(filepaths: List[str],
                 de_vocab: Vocab,
                 en_vocab: Vocab,
                 de_tokenizer: Any,
                 en_tokenizer: Any) -> List[Tuple[torch.Tensor]]:
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))

    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [de_vocab[token] for token in de_tokenizer(raw_de)],
            dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en)],
            dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))

    return data


def download() -> Tuple[List[Tuple[Union[torch.Tensor, Vocab]]]]:
    train_filepaths = [
        extract_archive(download_from_url(urllib.parse.urljoin(URL_BASE, url)))[0] for url in TRAIN_URLS
    ]
    val_filepaths = [
        extract_archive(download_from_url(urllib.parse.urljoin(URL_BASE, url)))[0] for url in VAL_URLS
    ]
    test_filepaths = [
        extract_archive(download_from_url(urllib.parse.urljoin(URL_BASE, url)))[0] for url in TEST_URLS
    ]

    def _spacy_tokenize(x, spacy):
        return [tok.text for tok in spacy.tokenizer(x)]

    de_tokenizer = partial(_spacy_tokenize, spacy=spacy.blank("de"))
    en_tokenizer = partial(_spacy_tokenize, spacy=spacy.blank("en"))

    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(
        train_filepaths,
        de_vocab,
        en_vocab,
        de_tokenizer,
        en_tokenizer
    )
    val_data = data_process(
        val_filepaths,
        de_vocab,
        en_vocab,
        de_tokenizer,
        en_tokenizer
    )
    test_data = data_process(
        test_filepaths,
        de_vocab,
        en_vocab,
        de_tokenizer,
        en_tokenizer
    )
    return train_data, val_data, test_data, de_vocab, en_vocab
