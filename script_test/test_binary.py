import numpy as np
import pandas as pd
import argparse


def bin(data):
    for i in range(len(data)):
        if i % 100000 == 0:
            print(i)
        if data['rating'][i] >= 4:
            data['rating'][i] = 1.
        else:
            data['rating'][i] = 0.
    return data

def main(dataset):
    path = 'data/' + dataset + '/'
    for t in ['train', 'test']:
        data = pd.read_csv(path + 'squ0_' + t + '.csv', header=None, names=['uid', 'iid', 'rating'])
        bi_data = bin(data)
        bi_data.to_csv(path + 'squ0b_' + t + '.csv', header=None, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1', help='name of dataset')
    args = parser.parse_args()
    
    main(args.dataset)
