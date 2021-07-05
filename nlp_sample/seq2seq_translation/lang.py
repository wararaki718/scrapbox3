from collections import defaultdict


SOS_TOKEN = 0
EOS_TOKEN = 1


class Lang:
    def __init__(self, name: str):
        self.name = name
        self.word2index = dict()
        self.word2count = defaultdict(int)
        self.index2word = ["SOS", "EOS"]
    
    @property
    def n_words(self) -> int:
        return len(self.index2word)

    def add_sentence(self, sentence: str):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
        self.word2count[word] += 1
