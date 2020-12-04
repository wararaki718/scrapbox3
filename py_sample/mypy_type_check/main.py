from typing import List, Tuple

from modules import f5, f6


def f1(x: int) -> List[int]: # ok
    return list(range(x))


def f2(x: int) -> Tuple[int]: # return type error
    return list(range(x))


def f3(x: float) -> List[int]: # argument type error
    return list(range(x))


def f4(x): # ok (if the function is not defined the type-hist, the function is ignored.)
    return list(range(x))


def main():
    a1 = f1(5)
    a2 = f2(5)
    a3 = f3(5)
    a4 = f4(5)
    a5 = f5(5)
    a6 = f6(5)

    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)
    print(a6)
    print('DONE')


if __name__ == '__main__':
    main()
