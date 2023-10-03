import random
import numpy as np
import time
from method.utils import computeRMSE
from method.scratch import Scratch

class Additive(Scratch):
    def __init__(self, param={}, groupNum=2, name='additive'):
        super(Additive, self).__init__(param, name)

        self.groupNum = groupNum
        self.epochs = self.epochs // self.groupNum
        self.userMats = []
        self.itemMats = []


    def learn(self, train_ratings, train_inds, group_index, test_ratings, test_inds, fix_seed=True, verbose=False):
        '''
        train_rating:   array   [userNum, itemNum]
        train_user:     list    [userNum]
        '''
        assert len(train_ratings) == self.groupNum and len(train_inds) == self.groupNum and len(group_index) == self.groupNum
        assert len(test_ratings) == self.groupNum and len(test_inds) == self.groupNum

        # Additive training
        self.userMats = []
        self.itemMats = []
        for i in range(self.groupNum):
            if i == 0:
                tmp_train_rating = train_ratings[i].copy()
                tmp_train_ind = train_inds[i].copy()
                tmp_train_user = group_index[i].copy()
                tmp_test_rating = test_ratings[i].copy()
                tmp_test_ind = test_inds[i].copy()
                userMat, itemMat = super(Additive, self).train(tmp_train_rating, tmp_train_ind, tmp_train_user, 
                                                                tmp_test_rating, tmp_test_ind, fix_seed, verbose=verbose)
                self.userMats.append(userMat)
                self.itemMats.append(itemMat)
            else:
                tmp_train_rating += train_ratings[i]
                tmp_train_ind += train_inds[i]
                tmp_train_user += group_index[i]
                tmp_test_rating += test_ratings[i]
                tmp_test_ind += test_inds[i]
                userMat, itemMat = super(Additive, self).train(tmp_train_rating, tmp_train_ind, tmp_train_user, 
                                                                tmp_test_rating, tmp_test_ind, fix_seed, True, self.userMats[-1], self.itemMats[-1], verbose=verbose)
                self.userMats.append(userMat)
                self.itemMats.append(itemMat)

        return userMat, itemMat


    def unlearn(self, group_index, del_train_ratings, del_train_inds, del_group_index, del_test_ratings, del_test_inds, del_user=[], del_point=[], fix_seed=True, verbose=False):
        '''
        train_rating:   array   [userNum, itemNum]
        train_user:     list    [userNum]
        '''
        assert len(del_train_ratings) == self.groupNum and len(del_train_inds) == self.groupNum and len(group_index) == self.groupNum
        assert len(del_test_ratings) == self.groupNum and len(del_test_inds) == self.groupNum

        # find deletion
        retrain_gid = self.groupNum
        for user in del_user:
            for i in range(self.groupNum):
                if user - 1 in group_index[i]:
                    gid = i
                    break
            retrain_gid = min(gid, retrain_gid)
        # additive retraining
        tmp_train_rating = del_train_ratings[0].copy()
        tmp_train_ind = del_train_inds[0].copy()
        tmp_del_train_user = del_group_index[0].copy()
        tmp_test_rating = del_test_ratings[0].copy()
        tmp_test_ind = del_test_inds[0].copy()
        for i in range(self.groupNum):
            if i != 0:
                tmp_train_rating += del_train_ratings[i]
                tmp_train_ind += del_train_inds[i]
                tmp_del_train_user += del_group_index[i]
                tmp_test_rating += del_test_ratings[i]
                tmp_test_ind += del_test_inds[i]
            if i >= retrain_gid:
                userMat, itemMat = super(Additive, self).train(tmp_train_rating, tmp_train_ind, tmp_del_train_user, 
                                                                tmp_test_rating, tmp_test_ind, fix_seed, True, self.userMats[i-1], self.itemMats[i-1], verbose=verbose)
                self.userMats[i] = userMat
                self.itemMats[i] = itemMat
        
        return userMat, itemMat


    def batchLearn(self, train_ratings, test_ratings, batch_size=4096000, fix_seed=True, verbose=False, file_type='txt'):
        '''
        train_rating:   list    [ratingNum] e.g. ['0,0,4', ...]
        '''
        assert len(train_ratings) == self.groupNum and len(test_ratings) == self.groupNum

        # Additive training
        self.userMats = []
        self.itemMats = []
        for i in range(self.groupNum):
            if i == 0:
                tmp_train_rating = train_ratings[i].copy()
                tmp_test_rating = test_ratings[i].copy()
                userMat, itemMat = super(Additive, self).batchTrain(tmp_train_rating, tmp_test_rating, batch_size, fix_seed, verbose=verbose, file_type=file_type)
                self.userMats.append(userMat)
                self.itemMats.append(itemMat)
            else:
                tmp_train_rating += train_ratings[i]
                tmp_test_rating += test_ratings[i]
                userMat, itemMat = super(Additive, self).batchTrain(tmp_train_rating, tmp_test_rating, batch_size, fix_seed, True, self.userMats[-1], self.itemMats[-1], verbose=verbose, file_type=file_type)
                self.userMats.append(userMat)
                self.itemMats.append(itemMat)

        return userMat, itemMat


    def batchUnlearn(self, group_index, del_train_ratings, del_test_ratings, batch_size=4096000, del_user=[], del_point=[], fix_seed=True, verbose=False, file_type='txt'):
        '''
        train_rating:   list    [ratingNum] e.g. ['0,0,4', ...]
        '''
        assert len(del_train_ratings) == self.groupNum and len(del_test_ratings) == self.groupNum

        # find deletion
        retrain_gid = self.groupNum
        for user in del_user:
            for i in range(self.groupNum):
                if user - 1 in group_index[i]:
                    gid = i
                    break
            if gid < retrain_gid:
                retrain_gid = gid
        # additive retraining
        tmp_train_rating = del_train_ratings[0].copy()
        tmp_test_rating = del_test_ratings[0].copy()
        for i in range(self.groupNum):
            if i != 0:
                tmp_train_rating += del_train_ratings[i]
                tmp_test_rating += del_test_ratings[i]
            if i >= retrain_gid:
                userMat, itemMat = super(Additive, self).batchTrain(tmp_train_rating, tmp_test_rating, batch_size, fix_seed, True, self.userMats[i-1], self.itemMats[i-1], verbose=verbose, file_type=file_type)
                self.userMats[i] = userMat
                self.itemMats[i] = itemMat
        
        return userMat, itemMat


