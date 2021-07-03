import glob
import os
from typing import Tuple
import unicodedata

from util import ALL_LETTERS


def unicode2ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in ALL_LETTERS
    )


def readlines(filename: str) -> list:
    lines = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            lines.append(unicode2ascii(line.strip()))
    return lines


def load(data_dir: str = "data/names/*.txt") -> Tuple[dict, list]:
    all_categories = list()
    category_lines = dict()
    for filename in glob.glob(data_dir):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        category_lines[category] = readlines(filename)
    return all_categories, category_lines
