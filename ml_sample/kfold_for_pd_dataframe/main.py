import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def main():
    df = pd.DataFrame(np.random.random((100, 4)))
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(df):
        tmp_df = df.take(train_index)
        print(tmp_df.shape)
    print('DONE')


if __name__ == '__main__':
    main()
