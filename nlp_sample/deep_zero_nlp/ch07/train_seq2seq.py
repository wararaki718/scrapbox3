import numpy as np

from dataset import sequence
from optimizer import Adam
from trainer import Trainer
from util import eval_seq2seq
from seq2seq import Seq2seq


def main() -> None:
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 128
    batch_size = 128
    max_epoch = 25
    max_grad = 5.0

    model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(1, max_epoch+1):
        trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

        correct_num = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)
        
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print(f'val acc {acc*100}%')
    print('DONE')


if __name__ == '__main__':
    main()
