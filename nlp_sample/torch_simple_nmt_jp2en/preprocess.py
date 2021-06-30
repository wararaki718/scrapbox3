import io
import os
from collections import Counter
from functools import partial
from typing import Any, List, Tuple, Union

import spacy
import torch
from torchtext.vocab import Vocab


DATA_DIR = ".data/data/orig/"
TRAIN_FILENAMES = [
    "kyoto-dev.ja",
    "kyoto-dev.en"
]
VAL_FILENAMES = [
    "kyoto-tune.ja",
    "kyoto-tune.en"
]
TEST_FILENAMES = [
    "kyoto-test.ja",
    "kyoto-test.en"
]


def build_vocab(filepath: str, tokenizer: Any) -> Vocab:
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for line in f:
            counter.update(tokenizer(line))
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process(filepaths: List[str],
                 ja_vocab: Vocab,
                 en_vocab: Vocab,
                 ja_tokenizer: Any,
                 en_tokenizer: Any) -> List[Tuple[torch.Tensor]]:
    raw_ja_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))

    data = []
    for (raw_ja, raw_en) in zip(raw_ja_iter, raw_en_iter):
        ja_tensor_ = torch.tensor(
            [ja_vocab[token] for token in ja_tokenizer(raw_ja)],
            dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en)],
            dtype=torch.long
        )
        data.append((ja_tensor_, en_tensor_))

    return data


def preprocessing() -> Tuple[List[Tuple[Union[torch.Tensor, Vocab]]]]:
    train_filepaths = [
        os.path.join(DATA_DIR, filename) for filename in TRAIN_FILENAMES
    ]
    val_filepaths = [
        os.path.join(DATA_DIR, filename) for filename in VAL_FILENAMES
    ]
    test_filepaths = [
        os.path.join(DATA_DIR, filename) for filename in TEST_FILENAMES
    ]

    def _spacy_tokenize(x, spacy):
        return [tok.text for tok in spacy.tokenizer(x)]

    ja_tokenizer = partial(_spacy_tokenize, spacy=spacy.load("ja_core_news_sm"))
    en_tokenizer = partial(_spacy_tokenize, spacy=spacy.blank("en"))

    ja_vocab = build_vocab(train_filepaths[0], ja_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(
        train_filepaths,
        ja_vocab,
        en_vocab,
        ja_tokenizer,
        en_tokenizer
    )
    val_data = data_process(
        val_filepaths,
        ja_vocab,
        en_vocab,
        ja_tokenizer,
        en_tokenizer
    )
    test_data = data_process(
        test_filepaths,
        ja_vocab,
        en_vocab,
        ja_tokenizer,
        en_tokenizer
    )
    return train_data, val_data, test_data, ja_vocab, en_vocab
