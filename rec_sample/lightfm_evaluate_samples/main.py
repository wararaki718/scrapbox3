from lightfm.datasets import movielens
import numpy as np

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

def main():
    data = fetch_movielens()

    # check dataset
    for key, value in data.items():
        print(key, type(value), value.shape)
    print()
    
    # get dataset
    train = data['train']
    test = data['test']

    # modeling
    model = LightFM(learning_rate=0.05, loss='bpr')
    model.fit(train, epochs=10)

    p_test = precision_at_k(model, test, k=10, train_interactions=train)
    r_test = recall_at_k(model, test, k=10, train_interactions=train)
    a_test = auc_score(model, test, train_interactions=train)

    print(p_test)
    print(r_test)
    print(a_test)
    print('DONE')


if __name__ == '__main__':
    main()
