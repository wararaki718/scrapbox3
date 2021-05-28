import re
from collections import Counter
from typing import List, Set


def words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def P(word: str) -> float:
    global WORDS
    global N
    return WORDS[word] / N


def correction(word: str) -> float:
    return max(candidates(word), key=P)


def candidates(word: str) -> bool:
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words: List[str]) -> Set[str]:
    global WORDS
    return set(word for word in words if word in WORDS)


def edits1(word: str) -> Set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str) -> Set[str]:
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def main():
    filename = "big.txt"
    global WORDS
    with open(filename, "rt") as f:
        WORDS = Counter(words(f.read()))

    global N
    N = sum(WORDS.values())

    print("# candidate model")
    target = "somthing"
    print(f"edits1({target})->{known(edits1(target))}")

    target = "something"
    print(f"edits2({target})->{known(edits2(target))}")
    target = "somthing"
    print(f"edits2({target})->{known(edits2(target))}")

    print("# language model")
    target = "the"
    print(f"P({target})->{P(target)}")
    target = "outrivaled"
    print(f"P({target})->{P(target)}")
    target = "unmentioned"
    print(f"P({target})->{P(target)}")

    print("# error model")
    target = "speling"
    print(f"{target}->{correction(target)}")
    target = "korrectud"
    print(f"{target}->{correction(target)}")

    print("DONE")


if __name__ == "__main__":
    main()
