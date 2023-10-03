from method.scratch import Scratch
from method.utils import baseTest
import numpy as np
import torch
from torch import nn

class Sisa(Scratch):
    def __init__(self, param={}, model_type='mf', n_group=5, group_index=[]):
        super(Sisa, self).__init__(param, model_type)

        self.n_group = n_group
        self.group_index = group_index
        self.epochs = self.epochs

        self.model_list = []


    def test(self, test_data, verbose, save_dir):
        rmse, ndcg, hr = baseTest(test_data, self.model_list, nn.MSELoss(reduction='sum'), self.device, verbose)
        log = {'total_rmse': rmse,
                'total_ndcg': ndcg,
                'total_hr': hr}
        np.save(save_dir + '/log0', log)

    def learn(self, train_dlist, test_dlist, test_data, verbose, save_dir):
        '''
        train_dlist:   list of dataloader[n_group]
        '''
        assert len(train_dlist) == self.n_group
        assert len(test_dlist) == self.n_group
        
        # sisa training
        for i in range(self.n_group):
            given_model = ''
            model = super(Sisa, self).train(train_dlist[i], test_dlist[i], test_data, verbose, save_dir, i+1, given_model)
            self.model_list.append(model)

        # merge user mat
        if self.model_type == "nmf":
            weight_list1 = [m.user_mat_mf.weight.to(self.device) for m in self.model_list]
            weight_list2 = [m.user_mat_mlp.weight.to(self.device) for m in self.model_list]
            merged_weight1 = torch.zeros_like(weight_list1[0])
            merged_weight2 = torch.zeros_like(weight_list2[0])

            for i in range(self.n_group):
                merged_weight1[ self.group_index[i] ] = weight_list1[i][ self.group_index[i] ]
                merged_weight2[ self.group_index[i] ] = weight_list2[i][ self.group_index[i] ]
            for m in self.model_list:
                m.user_mat_mf.weight = nn.Parameter(merged_weight1)
                m.user_mat_mlp.weight = nn.Parameter(merged_weight2)
        else:
            weight_list = [m.user_mat.weight.to(self.device) for m in self.model_list]
            merged_weight = torch.zeros_like(weight_list[0]).to(self.device)

            for i in range(self.n_group):
                merged_weight[ self.group_index[i] ] = weight_list[i][ self.group_index[i]]
            for m in self.model_list:
                m.user_mat.weight = nn.Parameter(merged_weight)

        # total test
        self.test(test_data, verbose, save_dir)

        return self.model_list


    def unlearn(self, model_list, train_dlist, test_dlist, test_data, del_user, verbose, save_dir):   
        '''
        train_dlist:   list of dataloader[n_group]
        '''
        self.model_list = model_list
        
        assert len(train_dlist) == self.n_group
        assert len(test_dlist) == self.n_group

        # find deletion
        retrain_gid = set()
        for user in del_user:
            for i in range(self.n_group):
                if user in self.group_index[i]:
                    retrain_gid.add(i)
                    break

        # sisa retraining
        model_before_unlearn = model_list[0]

        for i in retrain_gid:
            given_model = ''
            model = super(Sisa, self).train(train_dlist[i], test_dlist[i], test_data, verbose, save_dir, i+1, given_model)
            self.model_list[i] = model
        
        # merge user mat  
        if self.model_type == "nmf":
            weight_list1 = [m.user_mat_mf.weight.to(self.device) for m in self.model_list]
            weight_list2 = [m.user_mat_mlp.weight.to(self.device) for m in self.model_list]
            merged_weight1 = model_before_unlearn.user_mat_mf.weight.clone()
            merged_weight2 = model_before_unlearn.user_mat_mlp.weight.clone()
            
            for i in retrain_gid:
                merged_weight1[ self.group_index[i] ] = weight_list1[i][ self.group_index[i] ]
                merged_weight2[ self.group_index[i] ] = weight_list2[i][ self.group_index[i] ]
            
            for m in self.model_list:
                m.user_mat_mf.weight = nn.Parameter(merged_weight1)
                m.user_mat_mlp.weight = nn.Parameter(merged_weight2)

        else:
            weight_list = [m.user_mat.weight.to(self.device) for m in self.model_list]
            merged_weight = model_before_unlearn.user_mat.weight.clone()

            for i in retrain_gid:
                merged_weight[ self.group_index[i] ] = weight_list[i][ self.group_index[i]]
            for m in self.model_list:
                m.user_mat.weight = nn.Parameter(merged_weight)

        # total test
        self.test(test_data, verbose, save_dir)

        return self.model_list
