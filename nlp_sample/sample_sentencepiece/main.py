import sentencepiece as spm


def train():
    spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=model/sample --vocab_size=2000')


def test():
    sp = spm.SentencePieceProcessor()
    sp.load('model/sample.model')

    text = 'This is a test.'
    print('encode:')
    print(sp.encode_as_pieces(text))
    print(sp.encode_as_ids(text))
    print()

    print('decode:')
    print(sp.decode_pieces(['_This', '_is', '_a', '_t', 'est']))
    print(sp.decode_ids([209, 31, 9, 375, 586]))
    print()

    print(f'vocab size: {sp.get_piece_size()}')
    print(f'id to piece: {sp.id_to_piece(209)}')
    print(f'piece to id: {sp.piece_to_id("_This")}')
    print(f'unknow token: {sp.piece_to_id("__MUST_BE_UNKNOWN__")}')
    print()

    print('contorl sumbol:')
    for _id in range(3):
        print(sp.id_to_piece(_id), sp.is_control(_id))
    print()


def main():
    train()
    test()

    print('DONE')


if __name__ == '__main__':
    main()
