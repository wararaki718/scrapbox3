import logging


logging.basicConfig(level=logging.INFO)

def func(x) -> str:
    if x == 'c':
        return None
    return x

def main():
    b = func('app_b')
    logging.info(b)


if __name__ == '__main__':
    main()
