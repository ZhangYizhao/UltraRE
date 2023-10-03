import time
import pandas as pd
import numpy as np
import argparse
from functools import wraps
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

def squ_id(pdr, file_name):
    # squeeze uid
    uid_dict = {}
    for i, ori in enumerate(pdr['uid'].unique()):
        uid_dict[ori] = i
    pdr['uid'] = pdr['uid'].map(uid_dict)

    # squeeze iid
    iid_dict = {}
    for i, ori in enumerate(pdr['iid'].unique()):
        iid_dict[ori] = i
    pdr['iid'] = pdr['iid'].map(iid_dict)
    
    # save file
    pdr.sort_values(by=['uid', 'iid']).to_csv(file_name + '.csv', index=False)

    print('squeeze done!')
    return pdr

def filter_interaction(pdr, low):
    # rough filtering
    res = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    print('>>> rough user:')
    item_data = pdr.groupby('uid').agg('count')
    for i in tqdm(pdr['uid'].unique()):
        if item_data['iid'][i] >= 5:
            res = res.append(pdr[pdr['uid'].values == i], ignore_index=True)
    pdr = res

    # subtle filtering
    while True:
        n_filter = 0
        # drop item
        print('>>> item:')
        user_data = pdr.groupby('iid').agg(list)
        for i in tqdm(pdr['iid'].unique()):
            if len(user_data['uid'][i]) < low:
                pdr = pdr[pdr['iid'].values != i]
                n_filter += 1

        # drop user
        print('>>> user:')
        item_data = pdr.groupby('uid').agg(list)
        for i in tqdm(pdr['uid'].unique()):
            if len(item_data['iid'][i]) < low:
                pdr = pdr[pdr['uid'].values != i]
                n_filter += 1
        
        # terminate
        if n_filter == 0:
            break

    # reset index
    pdr.reset_index(drop=True, inplace=True)
    
    print('filtering done!')
    return pdr

def div(pdr, train_ratio):
    np.random.seed(0)
    train_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    test_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    for i in tqdm(range(len(pdr['uid'].unique()))):
        user_data = pdr[pdr['uid'] == i]
        total = len(user_data)

        n_train = int(total * train_ratio)
        train_idx = np.random.choice(total, n_train, replace=False)

        n_test = total - n_train
        test_idx = list(set(np.arange(total)) - set(train_idx))

        user_train = user_data.iloc[np.sort(train_idx), :]
        user_test = user_data.iloc[np.sort(test_idx), :]

        train_df = train_df.append(user_train, ignore_index=True)
        test_df = test_df.append(user_test, ignore_index=True)

    train_df['rating'] = np.float16(train_df['rating'])
    test_df['rating'] = np.float16(test_df['rating'])
    train_df.to_csv('squ0_train.csv', header=None, index=False)
    test_df.to_csv('squ0_test.csv', header=None, index=False)
    print('divide done!')

@timefn
def main(dataset):
    # read data
    pdr = pd.read_csv('data/' + dataset + '/dr_ratings.csv')

    # drop ratings
    pdr = filter_interaction(pdr, 5)
    pdr = squ_id(pdr, 'dr_ratings')

    # divide train : test (9:1)
    div(pdr, 0.9)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='name of dataset')
    args = parser.parse_args()
    
    main(args.dataset)