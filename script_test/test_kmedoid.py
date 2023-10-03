import numpy as np
import pandas as pd
import time
from functools import wraps
import warnings
from scipy.sparse import coo_matrix
import argparse
from sklearn.metrics.pairwise import pairwise_distances

def timefn(fn):
    '''compute time cost'''
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time

def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating
    ind = np.ones_like(val, dtype=int)

    # val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)  # set to float! int will cause error in kmeans

    return ind_mat.tocsr()

def sortArr(sim, order='asc'):
    '''
    Parameters
    ----------
    sim:        array [n_user, n_group]
    order:      str in ['asc', 'des']
    
    Returns
    -------
    zip(raw, col):  raw - list [n_user]
                    col - list [n_group]
    '''
    if order == 'asc':
        idx = np.argsort(sim, axis=None)
    else:
        idx = np.argsort(sim, axis=None)[::-1]
    raw, col = np.unravel_index(idx, sim.shape)
    return zip(raw, col)


def singleKmedoids(k, n_user, dist_arr, arr_type, balanced, max_iter):
    '''
    Parameters
    ----------
    dist_arr:   array of float / csc_matrix [n_user, n_user]

    Returns
    -------
    label:      array of int [n_user]
    inertia:    float
    '''
    label = np.zeros(n_user, dtype=int)
    group_len = int(np.ceil(n_user/k))
    # init
    cen_idx = np.random.choice(n_user, k, replace=False)
    if arr_type == 'sparse':
        max_dist = dist_arr.max()

    for i in range(max_iter):
        print('iter', i, end=': ')
        beg = time.time()
        if arr_type == 'dense':
            dist = dist_arr[:, cen_idx]
        else:
            dist = dist_arr[:, cen_idx].toarray()  # dense mat
            # replace zero with max
            dist[dist == 0] += (1 + max_dist)
        # kmeans
        if balanced == False:
            new_label = dist.argmin(axis=1)
        # balanced kmeans
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * k
            dist_zip = sortArr(dist)
            assigned = []
            for (user_idx, group_idx) in dist_zip:
                if len(assigned) == n_user:
                    break
                if user_idx in assigned:
                    continue
                if label_count[group_idx] > 0:
                    new_label[user_idx] = group_idx
                    assigned.append(user_idx)
                    label_count[group_idx] -= 1
        inertia = np.sum(dist[np.arange(n_user), new_label])

        if (new_label == label).all():
            print()
            break
        label = new_label

        # update centroid
        for i in range(k):
            cluster_idx = np.arange(n_user)[new_label == i]
            cluster_dist = dist_arr[cluster_idx, :]
            cluster_inertia = np.sum(cluster_dist, axis=1)
            cen_idx[i] = cluster_idx[cluster_inertia.argmin()]
        print('time', time.time() - beg)

    return label, inertia

def kmedoids(n_group, n_user, arr, arr_type, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    arr:    dist array [n_user, n_user]
    '''
    tmp_inertia = 1e10
    for i in range(n_init):
        print('init', i, '-------')
        label, inertia = singleKmedoids(n_group, n_user, arr, arr_type, balanced, max_iter)
        if inertia < tmp_inertia:
            tmp_inertia = inertia
            fin_label = label
    return fin_label
    
def transform(label, n_group, dir):
    res = []
    for idx in range(n_group):
        res.append([i for i, x in enumerate(label) if x == idx])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.save(dir, res)

    
def main(dataset, dist_type):
    assert dataset in ['ah', 'ad', 'db', 'ml1', 'ml20', 'ml25', 'netf']
    path = 'data/' + dataset + '/spdist_' + dist_type + '_' + dataset + '.npy'
    if dataset == 'ah':
        n_user = 38609
        n_item = 18534
        arr_type = 'sparse'
    elif dataset ==  'ad':
        n_user = 5541
        n_item = 3568
        arr_type = 'sparse'
    elif dataset == 'am':
        n_user = 123960
        n_item = 50052
        arr_type = 'sparse'
    elif dataset == 'db':
        n_user = 108662
        n_item = 28
        arr_type = 'sparse'
    elif dataset == 'ml1':
        n_user = 6040
        n_item = 3416
        arr_type = 'dense'
    elif dataset == 'ml20':
        n_user = 138493
        n_item = 26744
        arr_type = 'dense'
    elif dataset == 'ml25':
        n_user = 162541
        n_item = 59047
        arr_type = 'dense'
    elif dataset == 'netf':
        n_user = 480189
        n_item = 17770
        arr_type = 'dense'

    # read data
    if arr_type == 'dense':
        spdist = np.load(path)
    else:
        spdist = np.load(path, allow_pickle=True).item()

    # ratings = readSparseMat('data/' + dataset + '/squ0_train.csv', n_user, n_item)
    # dist = pairwise_distances(ratings, metric='euclidean', n_jobs=-1)
    # print('reading done')

    # replace zero with max
    if arr_type == 'dense':
        spdist[spdist == 0] += (1 + spdist.max())
        # dist[dist == 0] += (1 + dist.max())

    # kmedoids
    for k in [2, 4, 8, 16]:
        if dist_type == 'eu':
            name = '/val/euclidean-'
        elif dist_type == 'cos':
            name = '/val/cosine-'
        label = kmedoids(k, n_user, spdist, arr_type, False)
        transform(label, k, 'data/' + dataset + name + 'skm' + str(k))
        print(k, 'skm <<< END', end='\n\n')

        label = kmedoids(k, n_user, spdist, arr_type, True)
        transform(label, k, 'data/' + dataset + name + 'bskm' + str(k))
        print(k, 'bskm <<< END', end='\n\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='dataset name')
    parser.add_argument('--dist', type=str, default='eu', help='type of distance')
    args = parser.parse_args()

    main(args.dataset, args.dist)
    
