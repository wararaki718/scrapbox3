from typing import List


def f5(x: int) -> List[int]: # ok
    return list(range(x))


def f6(x: int) -> None: # return type error
    return list(range(x))
