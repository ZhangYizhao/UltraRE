import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import pairwise_distances
import argparse
from itertools import combinations


def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',', names=['uid', 'iid', 'rating'])

    row = ratings['uid'].astype(int).values
    col = ratings['iid'].astype(int).values
    val = ratings['rating'].astype(float).values / max_rating
    # ind = np.ones_like(val, dtype=int)

    val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    # ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)  # set to float! int will cause error in kmeans

    return val_mat.tocsr(), ratings.groupby('uid').agg(list)#, ind_mat.tocsr()

def plot(data, top):
    fig = plt.figure()
    # fig.set_size_inches(4, 3)
    # ax = plt.subplot(111)
    plt.pcolor(data, cmap='rainbow')
    plt.colorbar()
    # plt.xlabel(r'$\epsilon$')
    # plt.ylabel(r'$\eta$')
    # ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8])
    # ax.set_yticklabels([0, 2, 4, 6, 8, 10])
    plt.savefig('result' + str(top) + '.pdf', bbox_inches='tight')

def buildDict(rating, n_user):
    dist_dict = {}
    dist_array = np.zeros((n_user, n_user), dtype=np.float16)

    for i, (fst, sec) in enumerate(combinations(np.arange(n_user), 2)):
        if i % 50000000 == 0:
            print(i)
        loc1 = np.in1d(rating['iid'][fst], rating['iid'][sec])
        loc2 = np.in1d(rating['iid'][sec], rating['iid'][fst])
        val1 = np.array(rating['rating'][fst])[loc1]
        val2 = np.array(rating['rating'][sec])[loc2]

        dist = np.float32(np.sqrt(sum((val1 - val2)**2)))
        dist_dict[fst, sec] = dist
        dist_array[fst, sec] = dist
        dist_array[sec, fst] = dist
    np.save('dist_dict', dist_dict)
    return dist_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=1000, help='max plot user')
    args = parser.parse_args()

    dir = 'data/ml20/squ0_train.csv'
    n_user = 138493
    n_item = 26744

    val_mat, rating = readSparseMat(dir, n_user, n_item)
    
    # dist of all items
    # dist = pairwise_distances(val_mat, metric='euclidean', n_jobs=-1)
    # print('Number of zero:', np.count_nonzero(dist==0))  # output: 138493
    # top = args.top
    # plot(dist[:top, :top], top)

    # dist of items in common
    dist = buildDict(rating, n_user)
    for top in [10, 100, 1000]:
        plot(dist[:top, :top], top)