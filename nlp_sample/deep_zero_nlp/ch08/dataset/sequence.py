import os
from typing import Dict, Optional, Tuple

import numpy as np


id_to_char = {}
char_to_id = {}


def _update_vocab(txt: str) -> None:
    chars = list(txt)
    for _, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char

def load_data(filename: str='addition.txt', seed: int=42) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    if not os.path.exists(filepath):
        print(f'No file: {filename}')
        return None
    
    questions = []
    answers = []

    with open(filepath, 'r') as f:
        for line in f:
            idx = line.find('_')
            questions.append(line[:idx])
            answers.append(line[idx:-1])
    
    for question, answer in zip(questions, answers):
        _update_vocab(question)
        _update_vocab(answer)
    
    x = np.zeros((len(questions), len(questions[0])), dtype=int)
    t = np.zeros((len(answers), len(answers[0])), dtype=int)

    for i, question in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(question)]
    for i, answer in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(answer)]
    
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)

def get_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    return char_to_id, id_to_char
