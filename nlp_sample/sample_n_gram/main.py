from collections import defaultdict
from typing import Dict


def unigram(text: str) -> Dict[str, int]:
    result = defaultdict(int)
    for i in range(len(text)):
        result[text[i:i+1]] += 1
    return result


def bigram(text: str) -> Dict[str, int]:
    result = defaultdict(int)
    for i in range(len(text)-1):
        result[text[i:i+2]] += 1
    return result


def trigram(text: str) -> Dict[str, int]:
    result = defaultdict(int)
    for i in range(len(text)-2):
        result[text[i:i+3]] += 1
    return result


def show(result: Dict[str, int]):
    n = sum(result.values())
    for key, value in result.items():
        print(key, value, value/n)


def main():
    text = "きしゃのきしゃがきしゃできしゃする。"

    print("# unigram")
    show(unigram(text))

    print("# bigram")
    show(bigram(text))

    print("# trigram")
    show(trigram(text))

    print("DONE")


if __name__ == "__main__":
    main()
