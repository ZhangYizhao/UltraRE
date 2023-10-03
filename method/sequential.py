from method.scratch import Scratch
import numpy as np

class Sequential(Scratch):
    def __init__(self, param={}, model_type='mf', n_group=5, group_index=[]):
        super(Sequential, self).__init__(param, model_type)

        self.n_group = n_group
        self.group_index = group_index
        # self.epochs = self.epochs // self.n_group
        self.epochs = int(np.ceil(self.epochs/self.n_group))

        self.model_list = []


    def learn(self, train_dlist, test_dlist, test_data, verbose, save_dir):
        '''
        train_dlist:   list of dataloader[n_group]
        '''
        assert len(train_dlist) == self.n_group
        assert len(test_dlist) == self.n_group
        
        # sequential training
        for i in range(self.n_group):
            given_model = '' if i == 0 else self.model_list[-1]
            model = super(Sequential, self).train(train_dlist[i], test_dlist[i], test_data, verbose, save_dir, i+1, given_model)
            self.model_list.append(model)

        return self.model_list


    def unlearn(self, model_list, train_dlist, test_dlist, test_data, del_user, verbose, save_dir):   
        '''
        train_dlist:   list of dataloader[n_group]
        '''
        assert len(train_dlist) == self.n_group
        assert len(test_dlist) == self.n_group

        # find deletion
        retrain_gid = self.n_group
        for user in del_user:
            for i in range(self.n_group):
                if user in self.group_index[i]:
                    gid = i
                    break
            retrain_gid = min(gid, retrain_gid)

        # sequential retraining
        self.model_list = model_list[:retrain_gid]
        for i in range(retrain_gid, self.n_group):
            given_model = '' if i == 0 else self.model_list[-1]
            model = super(Sequential, self).train(train_dlist[i], test_dlist[i], test_data, verbose, save_dir, i+1, given_model)
            self.model_list.append(model)

        return self.model_list
