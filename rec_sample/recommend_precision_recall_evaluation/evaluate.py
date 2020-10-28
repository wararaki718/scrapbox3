import numpy as np
from sklearn.metrics import confusion_matrix


def main():
    y_actual = np.array([5, 4, 3, 5, 2])
    y_predict = np.array([5, 4, 4, 5, 3])

    cm = confusion_matrix(y_actual, y_predict)

    recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    precision = np.mean(np.diag(cm) / np.sum(cm, axis=1))

    print(f'recall    : {recall}')
    print(f'precision : {precision}')


if __name__ == '__main__':
    main()
