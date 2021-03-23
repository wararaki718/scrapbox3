import pickle

import numpy as np

from trainer import Trainer
from optimizer import Adam
from cbow import CBOW
from util import create_context_target
from dataset import ptb


def main() -> None:
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_context_target(corpus, window_size)

    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)
    # trainer.plot()

    word_vecs = model.word_vecs
    params = {
        'word_vecs': word_vecs.astype(np.float16),
        'word_to_id': word_to_id,
        'id_to_word': id_to_word
    }
    with open('cbow_params.pkl', 'wb') as f:
        pickle.dump(params, f, -1)


if __name__ == '__main__':
    main()
