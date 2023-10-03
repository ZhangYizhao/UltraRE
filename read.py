import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

def readRating(dir, n_user, max_rating=5, del_user=[], del_rating=[], n_group=1, group_index=[], sort='r'):
    '''
    Parameters
    ----------
        del_user:   list of int [0, 1, 2, ...]
        del_rating: 2d array [[uid, iid], ...]

    Returns
    -------
        rating_lists:   list [n_group] of array [3, n_rating]
        group_index:    list [n_group] of array [group_len]
    '''
    if len(group_index) == 0:
        # uniform grouping
        group_len = int(np.ceil(n_user / n_group))
        org_index = np.arange(n_user).tolist()

        if n_group == 1:
            group_index = [org_index]
        else:
            np.random.seed(0)
            np.random.shuffle(org_index)
            group_index = []
            for i in range(n_group):
                group_index.append(org_index[i*group_len : (i+1)*group_len])

    rating_lists = []

    ratings = pd.read_csv(dir, header=None, sep=',')

    # sort group index
    if sort in ['d', 'a']:
        for d in ['ml1m']:
            if d in dir:
                dataset = d
                break
        sorted_index = sort_group(order='a', group_index=group_index, var='count',
                                    ratings0=ratings[0], dataset=dataset)
        tmp_index = group_index.copy()
        group_index = []
        for i in sorted_index:
            group_index.append(tmp_index[i])

    del_user = set(del_user)
    del_rating = np.array(del_rating)

    for i in range(n_group):
        # delete user
        print(group_index[i])
        print(del_user)
        intersetion = set(group_index[i]) - del_user
        loc = np.in1d(ratings[0], list(intersetion))  
        del_user -= intersetion
        ratings_group = ratings[loc]

        # normalization
        ratings_group = ratings_group.values.T
        ratings_group[2] /= max_rating

        rating_lists.append(ratings_group)

    return rating_lists, group_index


def sort_group(order='a', group_index=[], var='count', ratings0=[], dataset='ml1'):
    assert var in ['count', 'density']
    n_group = len(group_index)

    if var == 'count':
        sort_value = []  # rating count
        for index in group_index:
            loc = np.in1d(ratings0, index)
            sort_value.append(loc.sum())
    elif var == 'density':
        # read data
        dist_file = 'data/' + dataset + '/spdist_eu_' + dataset + '.npy'
        if dataset == 'ml1':
            dist = np.load(dist_file, allow_pickle=True)
        elif dataset == 'ah':
            dist = np.load(dist_file, allow_pickle=True).item()
        
        # compute density
        sort_value = []  # density
        for index in group_index:
            cluster_dist = dist[index][:, index]
            if dataset == 'ml1':
                squ_dist = 1 / cluster_dist[cluster_dist != 0]
            elif dataset == 'ah':
                squ_dist = 1 / cluster_dist.data

            density = np.sum(squ_dist) / (2 * len(index))
            sort_value.append(density)
            
    if order == 'a':
        sorted_index = np.argsort(sort_value)
    else:
        sorted_index = np.argsort(sort_value)[::-1]
    return sorted_index
    
class RatingData(Dataset):
    def __init__(self, rating_array):
        super(RatingData, self).__init__()
        self.users = rating_array[0].astype(int)
        self.items = rating_array[1].astype(int)
        self.ratings = rating_array[2].astype(float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        return (torch.tensor(user, dtype=torch.long), 
                torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32))


def loadData(data, batch=30000, n_worker=24, shuffle=True):
    '''
    Parameters
    ----------
    data:   RatingData object 
    '''
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=n_worker)


def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating

    val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)

    return val_mat.tocsr() #, ind_mat.tocsr()

