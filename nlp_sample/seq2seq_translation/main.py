from evaluate import evaluate_randomly
from model import AttentionDecoder, Encoder
from preprocess import preprocessing
from train import train_iterators


HIDDEN_SIZE = 256


def main():
    input_lang, output_lang, pairs = preprocessing("eng", "fra")
    
    encoder = Encoder(input_lang.n_words, HIDDEN_SIZE).cuda()
    decoder = AttentionDecoder(HIDDEN_SIZE, output_lang.n_words).cuda()

    train_iterators(pairs, encoder, decoder, input_lang, output_lang, 1000, 100)
    
    evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)
    print("DONE")


if __name__ == "__main__":
    main()
