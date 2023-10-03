import sys
import getopt
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import time

import torch
from torch import nn

param = {
    'toy': {
        'dir': 'data/toy/0_',
        'n_user': 1509,
        'n_item': 2072
    },
    'ml20': {
        'dir': 'data/ml20/squ0_',
        'n_user': 138493,
        'n_item': 26744
    },
    'ml25': {
        'dir': 'data/ml25/squ0_',
        'n_user': 162541,
        'n_item': 59047
    },
    'netf': {
        'dir': 'data/netf/squ0_',
        'n_user': 480189,
        'n_item': 17770
    }
}

def read_rating(n_user, n_item, dir, var='test'):
    assert var in ['train', 'test']
    pdr = pd.read_csv(dir + var + '.csv', header=None, sep=',')
    inv = pdr.values.T
    row = inv[0].astype(int)
    col = inv[1].astype(int)
    val = inv[2] / 5
    ind = np.ones_like(val, dtype=int)
    val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item)).toarray()
    ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item)).toarray()
    return val_mat, ind_mat, len(pdr)

class PMFLoss(nn.Module):
    def __init__(self, lam_u=0.1, lam_i=0.1):
        super().__init__()
        self.lam_u = lam_u
        self.lam_i = lam_i

    def forward(self, rating, u_mat, i_mat, mask):
        pred = torch.matmul(u_mat, i_mat.t())
        diff = (rating - pred)**2
        err = torch.sum(diff * mask)

        u_reg = self.lam_u * torch.sum(u_mat.norm(dim=1))
        i_reg = self.lam_i * torch.sum(i_mat.norm(dim=1))
        
        return err + u_reg + i_reg

def main(argv):
    k, lr = 15, 0.001
    dataset, epochs = 'toy', 100

    try:
        opts, args = getopt.getopt(argv, 'd:e', ['dataset=', 'epoch='])
    except getopt.GetoptError:
        print('python main.py --dataset <ml20> --epoch <100>')
    for opt, arg in opts:
        if opt in ['-d', '--dataset']:
            assert arg in ['toy', 'ml20', 'ml25', 'netf']
            dataset = arg
        elif opt in ['-e', '--epoch']:
            assert int(arg) > 0
            epochs = int(arg)

    # CPU/GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # read training data
    n_user = param[dataset]['n_user']
    n_item = param[dataset]['n_item']
    dir = param[dataset]['dir']

    train_val, train_ind, train_num = read_rating(n_user, n_item, dir, 'train')
    train_rating = torch.from_numpy(train_val).to(device)
    train_mask = torch.from_numpy(train_ind).to(device)

    # initial parameters
    user_mat = torch.randn(n_user, k, requires_grad=True, device=device)
    user_mat.data.mul_(1/np.sqrt(k))

    item_mat = torch.randn(n_item, k, requires_grad=True, device=device)
    item_mat.data.mul_(1/np.sqrt(k))

    # train
    model = PMFLoss()
    opt = torch.optim.SGD([user_mat, item_mat], lr=lr)

    start = time.time()
    for t in range(epochs):
        opt.zero_grad()
        loss = model(train_rating, user_mat, item_mat, train_mask)
        loss.backward()
        opt.step()
        if t % 10 == 0:
            print(f'epoch: [{t+1:>3d}/{epochs:>3d}] loss: {np.sqrt(loss.item()/train_num):>7f}')
    print('total time:', time.time() - start)

    # test
    test_val, test_ind, test_num = read_rating(n_user, n_item, dir, 'test')
    test_rating = torch.from_numpy(test_val).to(device)
    test_mask = torch.from_numpy(test_ind).to(device)

    with torch.no_grad():
        pred = torch.matmul(user_mat, item_mat.t())
        diff = (test_rating - pred)**2
        err = torch.sum(diff * test_mask)
        loss = np.sqrt(err.item() / test_num)
    print('test rmse: ', loss)

    torch.save(model.state_dict(), 'result/test.pth')


if __name__ == '__main__':
    main(sys.argv[1:])