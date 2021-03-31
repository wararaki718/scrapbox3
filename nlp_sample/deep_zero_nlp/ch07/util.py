from typing import Any, Dict, List

import numpy as np


def clip_grads(grads: List[np.ndarray], max_norm: float) -> None:
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate


def eval_seq2seq(model: Any, question: np.ndarray, correct: np.ndarray, id_to_char: Dict[int, str], verbose: bool=False, is_reverse: bool=False) -> int:
    correct = correct.flatten()

    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbose:
        if is_reverse:
            question = question[::-1]
        
        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\33[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'X'
            print(f'{mark} {guess}')
        print('---')
    
    return int(guess == correct)
