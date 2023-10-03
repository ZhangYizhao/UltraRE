import numpy as np
import pandas as pd
from glob import glob


def main1(dataset):
    # read data
    pdr = pd.read_csv('data/' + dataset + '/raw_ratings.csv')

    # drop tag
    pdr = pdr.drop(columns=['timestamp'])

    # change name
    pdr = pdr.rename(columns={'userId': 'uid', 'movieId': 'iid'})

    # squeeze uid
    pdr['uid'] -= 1

    # squeeze iid
    cor_dict = {}
    for i, ori in enumerate(pdr['iid'].unique()):
        cor_dict[ori] = i
    pdr['iid'] = pdr['iid'].map(cor_dict)

    # save squeeze total
    pdr.sort_values(by=['uid', 'iid']).to_csv('data/' + dataset + '/ratings.csv', index=False)
    print('squeeze done!')

    # divide train : test (9:1)
    np.random.seed(0)
    train_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    test_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    for i in range(len(pdr['uid'].unique())):
        if i % 10000 == 0:
            print(i)
        user_data = pdr[pdr['uid'] == i]
        total = len(user_data)

        n_train = int(total * 0.9)
        train_idx = np.random.choice(total, n_train, replace=False)

        n_test = total - n_train
        test_idx = list(set(np.arange(total)) - set(train_idx))

        user_train = user_data.iloc[np.sort(train_idx), :]
        user_test = user_data.iloc[np.sort(test_idx), :]

        train_df = train_df.append(user_train, ignore_index=True)
        test_df = test_df.append(user_test, ignore_index=True)

    train_df['rating'] = np.float16(train_df['rating'])
    test_df['rating'] = np.float16(test_df['rating'])
    train_df.to_csv('data/' + dataset + '/squ0_train.csv', header=None, index=False)
    test_df.to_csv('data/' + dataset + '/squ0_test.csv', header=None, index=False)
    print('divide done!')

def main2():
    # read data
    data = []
    for iid, file_name in enumerate(glob('data/netf/training_set/*')):
        if iid % 1000 == 0:
            print(iid)
        with open(file_name, 'r') as f:
            f.readline()
            for line in f:
                rec = line.strip().split(',')
                uid = int(rec[0])
                rating = int(rec[1])
                data.append([uid, iid, rating])
    pdr = pd.DataFrame(data, columns=['uid', 'iid', 'rating'])
    pdr.sort_values(by=['uid', 'iid']).to_csv('data/netf/raw_ratings.csv', index=False, sep=',')
    print('reading done!')

    # squeeze uid (no need for squeeze iid)
    cor_dict = {}
    for i, ori in enumerate(pdr['uid'].unique()):
        cor_dict[ori] = i
    pdr['uid'] = pdr['uid'].map(cor_dict)

    # save squeeze total
    pdr.to_csv('data/netf/ratings.csv', index=False)
    print('squeeze done!')

    # divide train : test (9:1)
    np.random.seed(0)
    train_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    test_df = pd.DataFrame([], columns=['uid', 'iid', 'rating'])
    for i in range(len(pdr['uid'].unique())):
        if i % 10000 == 0:
            print(i)
        user_data = pdr[pdr['uid'] == i]
        total = len(user_data)

        n_train = int(total * 0.9)
        train_idx = np.random.choice(total, n_train, replace=False)

        n_test = total - n_train
        test_idx = list(set(np.arange(total)) - set(train_idx))

        user_train = user_data.iloc[np.sort(train_idx), :]
        user_test = user_data.iloc[np.sort(test_idx), :]

        train_df = train_df.append(user_train, ignore_index=True)
        test_df = test_df.append(user_test, ignore_index=True)

    train_df['rating'] = np.float16(train_df['rating'])
    test_df['rating'] = np.float16(test_df['rating'])
    train_df.to_csv('data/netf/squ0_train.csv', header=None, index=False)
    test_df.to_csv('data/netf/squ0_test.csv', header=None, index=False)
    print('divide done!')

def main3(dataset):
    assert dataset in ['ah', 'ab', 'ao']
    if dataset == 'ah':
        file_name = 'ratings_Health_and_Personal_Care'
    elif dataset == 'ab':
        file_name = 'ratings_Baby'
    elif dataset == 'ao':
        file_name = 'ratings_Office_Products'
    # read data
    pdr = pd.read_csv('data/' + dataset + '/' + file_name + '.csv', 
                        header=None, names=['uid', 'iid', 'rating', 'time'])

    # drop tag
    pdr = pdr.drop(columns=['time'])

    # squeeze uid and iid
    uid_dict = {}
    for i, ori in enumerate(pdr['uid'].unique()):
        uid_dict[ori] = i
    pdr['uid'] = pdr['uid'].map(uid_dict)
    
    iid_dict = {}
    for i, ori in enumerate(pdr['iid'].unique()):
        iid_dict[ori] = i
    pdr['iid'] = pdr['iid'].map(iid_dict)

    # save squeeze total
    pdr.sort_values(by=['uid', 'iid']).to_csv('data/' + dataset + '/ratings.csv', index=False)
    print('squeeze done!')
    
if __name__ == '__main__':
    # main1('ml20')
    # main1('ml25')
    # main2()  # 'netf'
    # main3('ah')
    main3('ab')
    main3('ao')
