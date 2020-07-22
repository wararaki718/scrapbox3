from typing import Callable


## definitions
f1: Callable[[int], int] = lambda x: x*1

def f2(x: int) -> int:
    return x*2

class C:
    @classmethod
    def f3(self, x: int) -> int:
        return x*3


def call_funcs(x: int, f: Callable[[int], int]) -> int:
    return f(x)


def main():
    x = 2
    print(call_funcs(x, f1))
    print(call_funcs(x, f2))
    print(call_funcs(x, C.f3))
    print('DONE')


if __name__ == '__main__':
    main()
