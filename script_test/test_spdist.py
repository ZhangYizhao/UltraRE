import multiprocessing as mp
import numpy as np
import time
import pandas as pd
from functools import wraps
import argparse
from itertools import combinations
from scipy.sparse.coo import coo_matrix
from tqdm import tqdm

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

def read_pd(dataset):
    data = pd.read_csv('data/' + dataset + '/squ0_train.csv',
                        header=None, names=['uid', 'iid', 'rating'])
    return data.groupby('uid').agg(list)

def compute_dist(tri, data, fst, n_user):
    for sec in range(fst+1, n_user):
        # find items in common
        loc1 = np.in1d(data['iid'][fst], data['iid'][sec])
        # early stop
        if loc1.any() == False:
            continue
        loc2 = np.in1d(data['iid'][sec], data['iid'][fst])
        val1 = np.array(data['rating'][fst])[loc1]
        val2 = np.array(data['rating'][sec])[loc2]
        # compute dist
        dist = np.float32(np.sqrt(sum((val1 - val2)**2)))

        # append shared list
        tri.append([fst, sec, dist])

def worker(tri, data, user_idx, n_user):
    for idx in user_idx:
        # compute front
        compute_dist(tri, data, idx, n_user)
        # compute back
        compute_dist(tri, data, n_user - 2 - idx, n_user)
    
@timefn
def multi(dataset, data):
    if dataset == 'ah':
        n_user = 38609
    elif dataset == 'ml1':
        n_user = 6040
    elif dataset == 'ml20':
        n_user = 138493 
    elif dataset == 'netf':
        n_user = 480189
    
    # multiprocess
    manager = mp.Manager()
    tri = manager.list()

    proc = []
    n_cpu = mp.cpu_count()
    user_idx = np.arange(int(n_user/2))  # work when n_user is odd number
    group_len = int(np.ceil(len(user_idx) / n_cpu))

    for i in range(n_cpu):
        proc.append(mp.Process(target=worker, 
                                args=(tri, data, 
                                        user_idx[i*group_len:(i+1)*group_len],
                                        n_user)))
        proc[-1].start()
        print(f'CPU [{i+1:>2d}/{n_cpu:>2d}] start!')
    for i, p in enumerate(proc):
        p.join()
        print(f'CPU [{i+1:>2d}/{n_cpu:>2d}] done!')
    
    # build mat and save
    tri = np.array(tri).T
    spdist = coo_matrix((tri[2], (tri[0], tri[1])), shape=(n_user, n_user), dtype=np.float32)
    spdist = spdist.toarray()
    spdist += spdist.T
    np.save('data/' + dataset + '/spdist_m_' + dataset, spdist)

@timefn
def single(dataset, dist_type, data):
    if dataset == 'ah':
        n_user = 38609
        arr_type = 'sparse'
    elif dataset == 'ad':
        n_user = 5541
        arr_type = 'sparse'
    elif dataset == 'am':
        n_user = 123960
        arr_type = 'sparse'
    elif dataset == 'db':
        n_user = 108662
        arr_type = 'sparse'
    elif dataset == 'ml1':
        n_user = 6040
        arr_type = 'dense'
    elif dataset == 'ml20':
        n_user = 138493
        arr_type = 'dense'
    elif dataset == 'netf':
        n_user = 480189
        arr_type = 'dense'

    # single process
    if arr_type == 'dense':
        spdist = np.zeros((n_user, n_user), dtype=np.float32)
    else:
        row = []
        col = []
        val = []

    for c, (i, j) in enumerate(combinations(np.arange(n_user), 2)):
        if c % 10000000 == 0:
            print(c)
        # find items in common
        loc1 = np.in1d(data['iid'][i], data['iid'][j])
        # early stop
        if loc1.any() == False:
            continue
        loc2 = np.in1d(data['iid'][j], data['iid'][i])
        val1 = np.array(data['rating'][i])[loc1]
        val2 = np.array(data['rating'][j])[loc2]
        # compute dist
        if dist_type == 'eu':
            dist = np.float32(np.sqrt(sum((val1 - val2)**2)))
        elif dist_type == 'cos':
            m1 = np.sqrt(sum(val1**2))
            m2 = np.sqrt(sum(val2**2))
            dist = np.float32(np.dot(val1, val2) / (m1 * m2))

        if arr_type == 'dense':
            spdist[i, j] = dist
        else:
            row.append(i)
            col.append(j)
            val.append(dist)
    
    # build array and save
    if arr_type == 'sparse':
        spdist = coo_matrix((val, (row, col)), shape=(n_user, n_user), dtype=np.float32)
        spdist = spdist.tocsc()
    spdist += spdist.T
    np.save('data/' + dataset + '/spdist_' + dist_type + '_' + dataset, spdist)


def main(dataset, dist_type):
    # read data
    data = read_pd(dataset)
    print('Reading done!')

    # multi(dataset, data)
    single(dataset, dist_type, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='name of dataset')
    parser.add_argument('--dist', type=str, default='eu', help='type of distance')
    args = parser.parse_args()
    
    main(args.dataset, args.dist)
