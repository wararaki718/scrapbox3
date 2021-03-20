from dataset import ptb


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print(f'corpus size: {len(corpus)}')
    print(f'corpus[:30]: {corpus[:30]}')
    print()

    print(f'id_to_word[0]: {id_to_word[0]}')
    print(f'id_to_word[1]: {id_to_word[1]}')
    print(f'id_to_word[2]: {id_to_word[2]}')
    print()

    print(f'word_to_id["car"]  : {word_to_id["car"]}')
    print(f'word_to_id["happy"]: {word_to_id["happy"]}')
    print(f'word_to_id["lexus"]: {word_to_id["lexus"]}')
    print('DONE')
