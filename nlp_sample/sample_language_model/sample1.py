N = 55161

words = {
    "おれ": 451,
    "は": 1643,
    "蕎麦": 15,
    "が": 1630,
    "大好き": 2,
    "で": 929,
    "ある": 279
}

def P(w: str) -> float:
    return words[w] / N

def main():
    # 1-gram
    p = 1.0
    for key in words.keys():
        print(f"P({key}): {P(key)}")
        p *= P(key)

    # P(text) = P(word_1) * P(word_2) * ... * P(word_n)
    print(f"P({list(words.keys())}): {p}")
    print("DONE")


if __name__ == "__main__":
    main()
