import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import argparse
from sklearn.metrics.pairwise import pairwise_distances

def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating
    ind = np.ones_like(val, dtype=int)

    # val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)  # set to float! int will cause error in kmeans

    return ind_mat.tocsr()

def main(dataset):
    assert dataset in ['ml1', 'ml20', 'ml25', 'netf']
    path = 'data/' + dataset + '/squ0_train.csv'
    if dataset == 'ah':
        n_user = 38609
        n_item = 18534
    elif dataset == 'ml1':
        n_user = 6040
        n_item = 3416
    elif dataset == 'ml20':
        n_user = 138493
        n_item = 26744
    elif dataset == 'ml25':
        n_user = 162541
        n_item = 59047
    elif dataset == 'netf':
        n_user = 480189
        n_item = 17770

    # read data
    data = readSparseMat(path, n_user, n_item)

    # compute dist
    dist = pairwise_distances(data, metric='euclidean', n_jobs=-1)

    # save data
    np.save('data/' + dataset + '/dist_' + dataset, dist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='dataset name')
    args = parser.parse_args()

    main(args.dataset)