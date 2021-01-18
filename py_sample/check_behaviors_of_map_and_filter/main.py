import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def f1(x: int) -> bool:
    logger.info(f'f1: {x}')
    return x%2 == 0

def f2(x: int) -> int:
    logger.info(f'f2: {x}')
    return x*x

def main():
    a = [1, 2, 3, 4]

    logger.info('result')
    b = filter(f1, a)
    logger.info('filtered')
    c = map(f2, b)
    logger.info('mapped')

    for x in c:
        logger.info(f'result: {x}')
    logger.info('DONE')

if __name__ == '__main__':
    main()
