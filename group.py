import numpy as np
from os.path import abspath, join, dirname, exists
import warnings

from method.utils import kmeans, kmedoids, ot_cluster
from method.utils import findNeighbor
from method.utils import lpa

DATA_DIR = abspath(join(dirname(__file__), 'data'))
SAVE_DIR = abspath(join(dirname(__file__), 'result'))


from sklearn.decomposition import PCA


class Group(object):
    def __init__(self, rating, dataset, user_mat=None):
        self.rating = rating  # csr_matrix
        # self.ind = ind
        self.dataset = dataset
        self.user_mat = user_mat
        self.n_user = self.rating.shape[0]
        self.n_item = self.rating.shape[1]

    def grouping(self, dataset='ml1m ', n_group=2, var='emb-ot', verbose=True):
        assert n_group > 1
        label_dir = DATA_DIR + '/' + dataset + '/val/' + var + str(n_group) + '.npy'

        # load label if it exists
        if exists(label_dir) == True:
            label = np.load(label_dir, allow_pickle=True)
            return label
        
        [trans_var, cluster_var] = var.strip().split('-')

        # embedding + clustering
        if cluster_var in ['ot']:
            #### embedding ####
            if trans_var == 'rating':
                embedding = self.rating
            elif trans_var == 'emb':
                embedding = self.user_mat       
            # OT_cluster            
            elif cluster_var == "ot":
                _, label = ot_cluster(embedding, n_group) 
        
        if verbose == True:
            # print result
            result = ''
            for i in range(n_group):
                num = str(i) + ': ' + str(len(np.where(label == i)[0])) + ', '
                result += num
            print(result)

        # transform
        res = []
        for idx in range(n_group):
            res.append([i for i, x in enumerate(label) if x == idx])
        
        # save label
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # print(type(res))
            np.save(label_dir, res)  # np.load(dir, allow_pickle=True)

        return label





