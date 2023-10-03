from os.path import exists
from os import mkdir
import numpy as np
import time
import warnings
import os

from read import readRating, loadData, readSparseMat
from read import RatingData
from method.utils import saveObject
from method.scratch import Scratch
from method.sisa import Sisa
from group import Group, DATA_DIR, SAVE_DIR


class InsParam(object):
    def __init__(self, dataset='toy', epochs=50, n_worker=24, layers=[32], n_group=2, del_per=2, del_type='test'):
        # model param
        self.k = 16  # dimension of embedding
        self.lam = 0.1  # regularization coefficient
        self.layers = layers  # structure of FC layers in DMF

        # training param
        self.seed = 42
        self.n_worker = n_worker
        self.batch = 3000 if dataset == 'toy' else 30000
        self.lr = 0.001
        self.lr_decay = 0.95
        self.momentum = 0.9 
        self.epochs = epochs
        self.n_group = n_group

        # dataset-varied param
        self.del_rating = []  # 2d array/list [[uid, iid], ...]
        self.dataset = dataset
        self.max_rating = 5
        self.del_per = del_per
        self.del_type = del_type
 
        if dataset == 'ml1m':
            self.train_dir = DATA_DIR + '/ml1m/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ml1m/squ0_test.csv'
            self.n_user = 6040
            self.n_item = 3416

            if self.del_type == 'rand':
                np.random.seed(0)
                n_del = int(self.del_per * self.n_user)
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
        
    def info(self):
        print(self.dataset, '-----------')
        print('Path of training data:', self.train_dir)
        print('Path of testing data:', self.test_dir)
        print('Number of users:', self.n_user)
        print('Number of items:', self.n_item)


class Instance(object):
    def __init__(self, param):
        self.param = param
        prefix = '/test/' if self.param.del_type == 'test' else '/' + str(self.param.del_per) + '/' + self.param.del_type + '/'
        self.name = prefix + self.param.dataset + '_g' + str(self.param.n_group) 
        # prefix = '/test/' if self.param.del_type == 'test' else '/' + str(self.param.del_per) + '/' + self.param.del_type + '/'
        # self.name = prefix + self.param.dataset + '_' + str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))#self.param.n_group)
        param_dir = SAVE_DIR + self.name
        if exists(param_dir) == False:
            os.makedirs(param_dir)

        # save param
        saveObject(param_dir + '/param', self.param)  # loadObject(dir + '/param')
        # save deletion
        deletion = [self.param.del_user, self.param.del_rating]
        arr = np.asarray(deletion, dtype = object)    
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            np.save(param_dir + '/deletion', arr)  # np.load('deletion.npy')

    # read raw data
    def __read(self, is_del=False, n_group=1, group_index=[]):
        del_user = self.param.del_user if is_del == True else []
        del_rating = self.param.del_rating if is_del == True else []

        train_rating, train_index = readRating(self.param.train_dir,
                                                self.param.n_user,
                                                self.param.max_rating,
                                                del_user, del_rating,
                                                n_group, group_index, 'a')
        # no deletion for testing data
        test_rating, _ = readRating(self.param.test_dir,
                                    self.param.n_user,
                                    self.param.max_rating,
                                    [], [],
                                    n_group, train_index)

        return train_rating, train_index, test_rating

    # sub function of self.runFull
    def __full(self, is_save, saving_name, model_type='mf', is_del=False, verbose=1):
        print(self.name, saving_name, 'begin:')
        # read raw data
        train_rating, _, test_rating = self.__read(is_del)

        # load data
        train_data = loadData(RatingData(train_rating[0]), self.param.batch, self.param.n_worker)
        test_data = loadData(RatingData(test_rating[0]), self.param.batch, self.param.n_worker, False)

        # mkdir
        if is_save == True:
            save_dir = SAVE_DIR + self.name + '/' + saving_name
            if exists(save_dir) == False:
                mkdir(save_dir)
        else:
            save_dir = ''

        # train model
        model = Scratch(self.param, model_type)
        model.train(train_data, test_data, [], verbose, save_dir)
        print('End of training', self.name, saving_name)
        print()

    # sub function of self.runGroup
    def __group(self, model_list, is_save, learn_type, saving_name, model_type='mf', 
                is_del=False, group_type='uniform', n_group=5, verbose=1):
        print(self.name, saving_name, 'begin:')
        # group
        if group_type == 'uniform':
            group_index = []
        else:
            val_mat = readSparseMat(self.param.train_dir, self.param.n_user, self.param.n_item)
            # load user mat for group dividing
            user_mat = np.load("{}/2/rand/ml1m_g0/MF_full_train/user_mat0.npy".format(SAVE_DIR), allow_pickle=True)
            start_time_stage1 = time.time()
            group_index = Group(val_mat, self.param.dataset, user_mat).grouping(self.param.dataset, 
                                                    n_group, group_type, 
                                                    verbose=False)

        # read raw data
        train_rating, train_index, test_rating = self.__read(is_del, n_group, group_index)
        
        # load data
        train_dlist, test_dlist = [], []
        if learn_type in ['sisa']:
            for i in range(n_group):
                if i == 0:
                    test_total = test_rating[i].copy()
                else:
                    test_total = np.hstack((test_total, test_rating[i]))
                train_dlist.append(loadData(RatingData(train_rating[i]), self.param.batch, self.param.n_worker))
                test_dlist.append(loadData(RatingData(test_rating[i]), self.param.batch, self.param.n_worker, False))
            test_data = loadData(RatingData(test_total), self.param.batch, self.param.n_worker, False)
        
        # mkdir
        if is_save ==True:
            save_dir = SAVE_DIR + self.name + '/' + saving_name
            if exists(save_dir) == False:
                mkdir(save_dir)
        else:
            save_dir = ''

        # train model
 
        if learn_type == 'sisa':
            model = Sisa(self.param, model_type, n_group, train_index)
        if is_del == False:
            model.learn(train_dlist, test_dlist, test_data, verbose, save_dir)
        else:
            del_user = self.param.del_user
            for rating in self.param.del_rating:
                if rating[0] not in del_user:
                    del_user.append(rating[0])
            model.unlearn(model_list, train_dlist, test_dlist, test_data, del_user, verbose, save_dir)

        return model.model_list


    #########################
    # gunFull, runGroup
    # calling function
    #########################
    
    def runFull(self, is_save=True, verbose=1):
        '''model MF'''
        # full train without deletion
        self.__full(is_save, 'MF_full_train', 'mf', False, verbose)

        # retrain from scratch after deletion
        self.__full(is_save, 'MF_retrain', 'mf', True, verbose)

    def runGroup(self, is_save=True, learn_type='seq', group_type='uniform', n_group=5, verbose=1):
        '''model MF'''
        # seq learn with deletion
        saving_name = 'MF_' + group_type + '_' + learn_type + '_learn'
        model_list = self.__group([], is_save, learn_type, saving_name, 'mf', 
                                    False, group_type, n_group, verbose)

        # seq unlearn with deletion
        saving_name = 'MF_' + group_type + '_' + learn_type + '_unlearn'
        self.__group(model_list, is_save, learn_type, saving_name, 'mf', 
                        True, group_type, n_group, verbose)

