import numpy as np
import time
from functools import wraps
import warnings
import argparse
import time

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


def singleLPA(n_group, n_user, dist_arr, balanced, n_neighbor, max_iter=10):
    '''
    Parameters
    ----------
    dist_arr:   2d-array [n_user, n_user]
    '''
    group_len = int(np.ceil(n_user/n_group))
    # init
    label = np.random.randint(0, n_group, size=(n_user))
    # aj_idx = np.argsort(dist_arr)[:, :n_neighbor]
    # aj_val = -1 * np.sort(dist_arr)[:, :n_neighbor]

    for i in range(max_iter):
        print('iter', i, end=': ')
        beg = time.time()
        # similarity to weight
        group_weight = np.zeros((n_user, n_group))
        for user_i in range(n_user):
            group_weight[:, label[user_i]] += np.exp(-dist_arr[user_i])
            # for nei_i in range(n_neighbor):
            #     group_idx = label[aj_idx[user_i, nei_i]]
            #     weight = np.exp(aj_val[user_i, nei_i])
            #     group_weight[user_i, group_idx] += weight

        # LPA
        if balanced == False:
            new_label = group_weight.argmax(axis=1)
        # BPLA
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * n_group
            sim_zip = sortArr(group_weight, 'des')
            assigned = []
            for (user_idx, group_idx) in sim_zip:
                if len(assigned) == n_user:
                    break
                if user_idx in assigned:
                    continue
                if label_count[group_idx] > 0:
                    new_label[user_idx] = group_idx
                    assigned.append(user_idx)
                    label_count[group_idx] -= 1
        inertia = np.sum(group_weight[np.arange(n_user), new_label])

        if (new_label == label).all():
            break
        label = new_label
        print('time', time.time() - beg)
    return label, inertia

def lpa(n_group, n_user, dist_arr, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    dist_arr:   2d-array [n_user, n_user]
    '''
    tmp_inertia = 1e10
    for i in range(n_init):
        print('init', i, '-------')
        label, inertia = singleLPA(n_group, n_user, dist_arr, n_user, balanced, max_iter)
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

    
def main(dataset):
    assert dataset in ['ah', 'ml1', 'ml20', 'ml25', 'netf']
    path = 'data/' + dataset + '/spdist' + dataset + '.npy'
    if dataset == 'ah':
        n_user = 38609
    elif dataset == 'ml1':
        n_user = 6040
    elif dataset == 'ml20':
        n_user = 138493
    elif dataset == 'ml25':
        n_user = 162541
    elif dataset == 'netf':
        n_user = 480189

    # read data
    dist_arr = np.load(path)

    # replace zero with max
    dist_arr[dist_arr == 0] += (1 + dist_arr.max())

    # LPA
    for k in [2, 3, 4, 5, 8]:
        label = lpa(k, n_user, dist_arr, False)
        transform(label, k, 'data/' + dataset + '/val/euclidean-lpa' + str(k))
        print(k, 'lpa <<< END', end='\n\n')

    k = 2
    label = lpa(k, n_user, dist_arr, True)
    transform(label, k, 'data/' + dataset + '/val/euclidean-blpa' + str(k))
    print(k, 'blpa <<< END', end='\n\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml', help='dataset name')
    args = parser.parse_args()

    main(args.dataset)
    
