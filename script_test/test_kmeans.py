import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import warnings
import time
from sklearn.metrics.pairwise import pairwise_distances

def readSparseMat(dir, n_user, n_item, max_rating=5, var='val'):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating
    if var == 'ind':
        ind = np.ones_like(val)
        mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    elif var == 'val':
        mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    
    return mat.tocsr()


# build ascending list
def buildArr(sim, order='asc'):
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


def singleKmeans(k, n_user, sp_mat, dist_type, balanced, max_iter):
    '''
    Parameters
    ----------
    sp_mat:     csr_matrix [n_user, n_embedding]

    Returns
    -------
    label:      array of int [n_user]
    inertia:    float
    '''
    label = np.zeros(n_user, dtype=int)
    group_len = 1 + n_user // k
    # init
    cen_idx = np.random.choice(n_user, k, replace=False)
    centroid = sp_mat[cen_idx].copy()

    for i in range(max_iter):
        print('iter', i, end=': ')
        beg = time.time()
        dist = pairwise_distances(sp_mat, centroid, metric=dist_type, n_jobs=-1)

        # kmeans
        if balanced == False:
            new_label = dist.argmin(axis=1)
        # balanced kmeans
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * k
            dist_zip = buildArr(dist)
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for j in range(k):
                centroid[j] = csr_matrix(sp_mat[label == j].mean(axis=0))
        print('time', time.time() - beg)
    return label, inertia

def kmeans(n_group, n_user, sp_mat, dist_type, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    sp_mat:     csr_matrix [n_user, n_embedding]
    '''
    tmp_inertia = 1e10
    for i in range(n_init):
        print('init', i, '-------')
        label, inertia = singleKmeans(n_group, n_user, sp_mat, dist_type, balanced, max_iter)
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


def main(args):
    if args.dataset == 'ah':
        n_user = 38609
        n_item = 18534
    elif args.dataset == 'ad':
        n_user = 5541
        n_item = 3568
    elif args.dataset == 'am':
        n_user = 123960
        n_item = 50052
    elif args.dataset == 'db':
        n_user = 108662
        n_item = 28
    elif args.dataset == 'ml1':
        n_user = 6040
        n_item = 3416
    elif args.dataset == 'ml20':
        n_user = 138493
        n_item = 26744
    elif args.dataset == 'netf':
        n_user = 480189
        n_item = 17770
    data_dir = 'data/' + args.dataset + '/'
    
    mat = readSparseMat(data_dir + 'squ0_train.csv', n_user, n_item, var=args.read)

    # kmeans
    if args.dist == 'eu':
        dist_type = 'euclidean'
    elif args.dist == 'cos':
        dist_type = 'cosine'

    for k in [2, 4, 8, 16]:
        label = kmeans(k, n_user, mat, dist_type, False)
        transform(label, k, data_dir + args.read + '/' + dist_type + '-km' + str(k))
        print(k, 'km <<< END', end='\n\n')

        label = kmeans(k, n_user, mat, dist_type,  True)
        transform(label, k, data_dir + args.read + '/' + dist_type + '-bkm' + str(k))
        print(k, 'bkm <<< END', end='\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='name of dataset')
    parser.add_argument('--read', type=str, default='val', help='type of read data')
    parser.add_argument('--dist', type=str, default='eu', help='type of distance')
    args = parser.parse_args()

    main(args)