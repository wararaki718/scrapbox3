import logging


logging.basicConfig(level=logging.INFO)

def func(x) -> str:
    if x == 'c':
        return None
    return x

def main():
    a = func('app_a')
    logging.info(a)


if __name__ == '__main__':
    main()
