from transformers import BertJapaneseTokenizer, BertModel


def main():
    model_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    inputs = tokenizer("私はカレーライスを食べます。", return_tensors='pt')
    print(inputs)

    outputs = model(**inputs)
    print(outputs.last_hidden_state)
    print(outputs.last_hidden_state.shape)
    print('DONE')


if __name__ == '__main__':
    main()
