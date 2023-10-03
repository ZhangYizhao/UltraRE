import numpy as np
import pandas as pd
import time
import argparse

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--dataset', type=str, default='toy', help='dataset name')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--batch', type=int, default=30000, help='number of batch size')
parser.add_argument('--verbose', type=int, default=1, help='verbose type')
parser.add_argument('--model', type=str, default='mf', help='name of model')

def seed_all(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# format data
def read_rating(dir, var='test'):
    assert var in ['train', 'test']
    pdr = pd.read_csv(dir + var + '.csv', header=None, sep=',')
    inv = pdr.values.T
    users = inv[0].astype(int)
    items = inv[1].astype(int)
    ratings = inv[2].astype(float) / 5
    return users, items, ratings

class ratingData(Dataset):
    def __init__(self, users, items, ratings):
        super().__init__()
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        return (torch.tensor(user, dtype=torch.long), 
                torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float))

def get_data(dataset, var, batch):
    dir = param[dataset]['dir']

    users, items, ratings = read_rating(dir, var)
    data = ratingData(users, items, ratings)
    return DataLoader(data, batch_size=batch, shuffle=True, num_workers=24)

# build model MF
class MF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(MF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(1)

# build model DMF
class DMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layers=[32]):
        super(DMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.layers = [k]
        for layer in layers:
            self.layers.append(layer)
        self.user_fc = nn.ModuleList()
        self.item_fc = nn.ModuleList()
        self.cos = nn.CosineSimilarity()

        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.user_fc.append(nn.Linear(in_size, out_size))
            self.item_fc.append(nn.Linear(in_size, out_size))
            self.user_fc.append(nn.ReLU())
            self.item_fc.append(nn.ReLU())
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()


    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        for i in range(len(self.user_fc)):
            user_embedding = self.user_fc[i](user_embedding)
            item_embedding = self.item_fc[i](item_embedding)
        rating = self.cos(user_embedding, item_embedding)
        return rating.squeeze()

# train
def train(dataloader, model, loss_fn, is_rmse, opt, device, verbose):
    size = len(dataloader.dataset)
    train_loss = 0
    rmse_loss = 0
    if is_rmse == False:
        rmse_fn = nn.MSELoss(reduction='sum')

    for batch, (user, item, rating) in enumerate(dataloader):
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)

        pred = model(user, item)
        loss = loss_fn(pred, rating)
        train_loss += loss.item()

        if is_rmse == False:
            with torch.no_grad():
                rmse = rmse_fn(pred, rating)
                rmse_loss += rmse.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if is_rmse == False:
            loss_val = loss.item() / len(user)
            rmse_val = np.sqrt(rmse.item() / len(user))
        else:
            loss_val = np.sqrt(loss.item() / len(user))
            rmse_val = loss_val

        if verbose == 2 and batch % 100 == 0:
            cur_batch = batch * len(user)
            print(f'Loss: {loss_val:>7f}, RMSE: {rmse_val:>7f} [{cur_batch:>8d}/{size:>8d}]')

    if is_rmse == False:
        train_loss /= size
        rmse_loss = np.sqrt(rmse_loss / size)
    else:
        train_loss = np.sqrt(train_loss / size)
        rmse_loss = train_loss
        
    return train_loss, rmse_loss

# test
def test(dataloader, model, loss_fn, device, verbose):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for user, item, rating in dataloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            pred = model(user, item)
            test_loss += loss_fn(pred, rating).item()
        test_loss = np.sqrt(test_loss / size)
    if verbose == 2:
        print(f'Test RMSE: {test_loss:>7f}')
    return test_loss

def main():
    k, lr = 15, 0.001

    args = parser.parse_args()

    assert args.dataset in ['toy', 'ml20', 'ml25', 'netf']
    dataset = args.dataset

    assert args.epoch > 0
    epochs = args.epoch

    assert args.batch > 0
    batch = args.batch

    assert args.verbose in [1, 2]
    verbose = args.verbose

    assert args.model in ['mf', 'dmf']
    model_type = args.model

    # seeding for reproducibility
    seed_all(args.seed)

    # get CPU/GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # read data
    train_data = get_data(dataset, 'train', batch)
    test_data = get_data(dataset, 'test', batch)

    # param setting
    n_user = param[dataset]['n_user']
    n_item = param[dataset]['n_item']

    if model_type == 'mf':
        model = MF(n_user, n_item, k).to(device)
        loss_fn = nn.MSELoss(reduction='sum')
        is_rmse = True
    elif model_type == 'dmf':
        model = DMF(n_user, n_item, k).to(device)
        loss_fn = nn.BCELoss()
        is_rmse = False

    rmse_fn = nn.MSELoss(reduction='sum')
        
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.95)

    # main loop
    for t in range(epochs):
        if verbose == 2:
            print(f'Epoch: [{t+1:>3d}/{epochs:>3d}] --------------------')
        epoch_start = time.time()
        train_loss, rmse_loss = train(train_data, model, loss_fn, is_rmse, opt, device, verbose)
        scheduler.step()  # lr decay
        test_loss = test(test_data, model, rmse_fn, device, verbose)
        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start))
        if verbose == 2:
            print('Time:', epoch_time)
        elif verbose == 1:
            print(f'Epoch: [{t+1:>3d}/{epochs:>3d}] train loss: {train_loss:>7f}, train RMSE: {rmse_loss:>7f}, test RMSE: {test_loss:>7f}, time:', epoch_time)
    

if __name__ == '__main__':
    main()