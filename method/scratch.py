import numpy as np
import time
from torch import nn
from torch import optim
import torch

from method.utils import seed_all, baseTrain, baseTest
from method.utils import MF


class Scratch(object):
    def __init__(self, param, model_type):
        # model param
        self.n_user = param.n_user
        self.n_item = param.n_item
        self.k = param.k
        self.lam = param.lam
        self.model_type = model_type

        # training param
        self.seed = param.seed
        self.lr = param.lr
        self.lr_decay = param.lr_decay
        self.momentum = param.momentum
        self.epochs = param.epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dis_type = param.dis_type
        if self.dis_type == 'nor':
            self.attr = []
        else:
            self.attr = param.attr


        # log
        self.log = {'train_loss': [], 
                    'test_rmse': [], 
                    'test_ndcg': [], 
                    'test_hr': [],  
                    'total_rmse': [],
                    'total_ndcg': [],
                    'total_hr': [],
                    'time': []}

        if self.model_type == 'mf':
            self.loss_fn = nn.MSELoss(reduction='sum')
            self.is_rmse = True
            # self.loss_fn = nn.BCELoss()
            # self.is_rmse = False


    def train(self, train_data, test_data, test_total=[], verbose=1, save_dir='', id=0, given_model=''):
        print('Using device:', self.device)
        # seed for reproducibility
        seed_all(self.seed)
        
        # build model
        if given_model == '':
            if self.model_type == 'mf':
                model = MF(self.n_user, self.n_item, self.k).to(self.device)
        else:
            model = given_model.to(self.device)

        # set optimizer
        if self.model_type == 'mf':
            opt = optim.SGD(model.parameters(),
                            lr=self.lr,
                            weight_decay=self.lam,
                            momentum=self.momentum)
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=self.lr_decay)

        # main loop
        for t in range(self.epochs):
            if verbose == 2:
                print(f'Epoch: [{t+1:>3d}/{self.epochs:>3d}] --------------------')
            epoch_start = time.time()

            # train
            train_loss, train_rmse = baseTrain(train_data, model, self.loss_fn, self.is_rmse, opt, self.device, verbose, self.dis_type, self.attr)
            if self.model_type == 'mf':
                scheduler.step()  # lr decay

            # test
            if self.__class__.__name__ == 'Sisa':
                models = self.model_list + [model]
            else:
                models = [model]

            # group test
            test_rmse, test_ndcg, test_hr = baseTest(test_data, models, nn.MSELoss(reduction='sum'), self.device, verbose)

            # total test
            if test_total == []:
                total_rmse = test_rmse
                total_ndcg = test_ndcg
                total_hr = test_hr
            else:
                total_rmse, total_ndcg, total_hr = baseTest(test_total, models, nn.MSELoss(reduction='sum'), self.device, verbose)

            # print info
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start))
            if verbose == 2:
                print('Time:', epoch_time)
            elif verbose == 1:# and t == self.epochs - 1:
                if test_total == []:
                    print(f'Epoch: [{t+1:>2d}/{self.epochs:>2d}]' + 
                        f' train loss: {train_loss:>.9f},' + 
                        f' train RMSE: {train_rmse:>.4f},' +
                        f' test RMSE: {test_rmse:>.4f},' + 
                        f' time:', epoch_time)
                else:
                    print(f'Epoch: [{t+1:>2d}/{self.epochs:>2d}]' + 
                        f' train loss: {train_loss:>.9f},' + 
                        f' train RMSE: {train_rmse:>.4f},' +
                        f' test RMSE: {test_rmse:>.4f},' + 
                        f' total RMSE: {total_rmse:>.4f},' + 
                        f' time:', epoch_time)


            # save log
            self.log['train_loss'].append(train_loss)
            self.log['test_rmse'].append(test_rmse)
            self.log['test_ndcg'].append(test_ndcg)
            self.log['test_hr'].append(test_hr)
            self.log['time'].append(epoch_time)
            if test_total != []:
                self.log['total_rmse'].append(total_rmse)
                self.log['total_ndcg'].append(total_ndcg)
                self.log['total_hr'].append(total_hr)

        # save model
        if len(save_dir) > 0:
            torch.save(model.state_dict(), save_dir + '/model' + str(id) + '.pth')
            # load torch model
            # model = MF(self.n_user, self.n_item)
            # model.load_state_dict(torch.load('model.pth'))

            if self.model_type in ['mf']:
                np.save(save_dir + '/user_mat' + str(id), model.user_mat.cpu().weight.detach().numpy())
                np.save(save_dir + '/item_mat' + str(id), model.item_mat.cpu().weight.detach().numpy())
            # load mat
            # user_mat = np.load('user_mat.npy')
            # item_mat = np.load('item_mat.npy')
            
            np.save(save_dir + '/log' + str(id), self.log)
            # load log
            # log = np.load('log.npy', allow_pickle=True).item()

        return model
