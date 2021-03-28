from util import eval_perplexity
from optimizer import SGD
from trainer import RnnlmTrainer
from dataset import ptb
from rnnlm import Rnnlm


def main():
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100
    time_size = 35
    lr = 20.0
    #max_epoch = 4
    max_epoch = 1
    max_grad = 0.25

    corpus, word_to_id, _ = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
    model.reset_state()

    ppl_test = eval_perplexity(model, corpus_test)
    print(f'test perplexity: {ppl_test}')

    model.save_params()
    print('DONE')


if __name__ == '__main__':
    main()
