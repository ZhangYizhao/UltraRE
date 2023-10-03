import numpy as np
from os.path import abspath, join, dirname, exists
import warnings

from method.utils import kmeans, kmedoids, ot_cluster
from method.utils import findNeighbor
from method.utils import lpa

DATA_DIR = abspath(join(dirname(__file__), 'data'))
SAVE_DIR = abspath(join(dirname(__file__), 'result'))

# import numpy as np
# from itertools import combinations
# from numpy.random import triangular
# from scipy import stats
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA


class Group(object):
    def __init__(self, rating, dataset, user_mat=None):
        self.rating = rating  # csr_matrix
        # self.ind = ind
        self.dataset = dataset
        self.user_mat = user_mat
        self.n_user = self.rating.shape[0]
        self.n_item = self.rating.shape[1]

        # self.user = []
        # self.rw = []
        # self.nm = []


    # def buildAj(self, var='pearson'):
    #     # transform UI matrix to User adjacency matrix
    #     user_aj = np.zeros((self.n_user, self.n_user), dtype=np.float16)

    #     # if [i, j] have rating on same item, aj[i, j] += 1
    #     if var == 'bool':
    #         for ind in self.ind.T:
    #             have_rating = np.where(ind == 1)[0]
    #             for comb in combinations(have_rating, 2):
    #                 user_aj[comb] += 1
    #         user_aj += user_aj.T

    #     # aj[i, j] += (max - |rating_i - rating_j|)
    #     elif var == 'float':
    #         max_rating = 1
    #         combs = combinations(range(self.n_user), 2)
    #         for rating in self.rating.T:
    #             for comb in combs:
    #                 user_aj[comb] += max_rating - np.abs(rating[comb[0]] - rating[comb[1]])
    #         user_aj += user_aj.T

    #     # regard user's rating on all items as embedding, compute EU dist
    #     elif var == 'euclidean':
    #         user_aj = -euclidean_distances(self.rating)

    #     # compute cosine dist
    #     elif var == 'cosine':
    #         user_aj = -cosine_distances(self.rating)

    #     # compute manhattan dist
    #     elif var == 'manhattan':
    #         user_aj = -manhattan_distances(self.rating)
        
    #     # pearson similarity
    #     elif var == 'pearson':
    #         combs = combinations(range(self.n_user), 2)
    #         for comb in combs:
    #             # intersection - implement 1
    #             # have_rating1 = np.where(self.ind[comb[0]] == 1)[0]
    #             # have_rating2 = np.where(self.ind[comb[1]] == 1)[0]
    #             # common_rating = list(set(have_rating1).intersection(set(have_rating2)))
    #             # intersection - implement 2
    #             # common_ind = self.ind[comb[0]] * self.ind[comb[1]]
    #             # common_rating = np.where(common_ind == 1)[0]
    #             # if len(common_rating) < 2:
    #             #     similarity = 0
    #             # else:
    #             #     # intersection
    #             #     rating1 = self.rating[comb[0]][common_rating]
    #             #     rating2 = self.rating[comb[1]][common_rating]
    #             #     similarity = stats.pearsonr(rating1, rating2)[0]
    #             # all items
    #             if len(np.where(self.ind[comb[0]] == 1)[0]) == 0 or len(np.where(self.ind[comb[1]] == 1)[0]) == 0:
    #                 similarity = 0
    #             else:
    #                 rating1 = self.rating[comb[0]]
    #                 rating2 = self.rating[comb[1]]
    #                 similarity = stats.pearsonr(rating1, rating2)[0]
    #             user_aj[comb] = similarity
    #         user_aj += user_aj.T
    #     # self.user = user_aj
    #     return user_aj


    def grouping(self, dataset='ml1m ', n_group=2, var='emb-ot', verbose=True):
        assert n_group > 1
        label_dir = DATA_DIR + '/' + dataset + '/val/' + var + str(n_group) + '.npy'

        # load label if it exists
        if exists(label_dir) == True:
            label = np.load(label_dir, allow_pickle=True)
            return label
        
        [trans_var, cluster_var] = var.strip().split('-')
        # community detection
        if cluster_var in ['lpa', 'blpa']:
            assert trans_var in ['bool', 'float', 'euclidean', 'pearson', 'cosine', 'manhattan']
            cache_dir = DATA_DIR + '/' + self.dataset +'/'
            nei_idx, nei_val = findNeighbor(cache_dir, self.rating, self.n_user, trans_var)
            # Label Propagation Algorithm (LPA)
            if cluster_var == 'lpa':
                label = lpa(n_group, self.n_user, nei_idx, nei_val)
            # Balanced LPA
            elif cluster_var == 'blpa':
                label = lpa(n_group, self.n_user, nei_idx, nei_val, True)
        
        # embedding + clustering
        elif cluster_var in ['kmeans', 'bekm', 'pca', 'skm', 'bskm', 'ot']:
            #### embedding ####
            if trans_var == 'rating':
                embedding = self.rating
            elif trans_var == 'emb':
                embedding = self.user_mat
            # elif trans_var == 'neighbor_mean':
            #     if len(self.nm) == 0:
            #         embedding = np.zeros((self.n_user, self.n_user), dtype=np.float16)
            #         for i in range(self.n_user):
            #             have_rating = np.where(self.ind[i] == 1)[0]
            #             if len(have_rating) > 0:
            #                 embedding[i] = np.mean(self.rating.T[have_rating], axis=0)
            #         self.nm = embedding
            #     else:
            #         embedding = self.nm
            # elif trans_var == 'random_walk':
            #     embedding = self.randomWalk()
            # elif trans_var == 'deep_walk':
            #     embedding = self.randomWalk(var='all')

            #### clustering ####
            # k-means
            if cluster_var == 'kmeans':
                _, label = kmeans(embedding, n_group, False)

            # Balanced Embedding k-mean (BEKM)
            elif cluster_var == 'bekm':
                _, label = kmeans(embedding, n_group, True)
                # label = kmeans(n_group, self.n_user, embedding, True)

            # Sparse k-medoids
            elif cluster_var == 'skm':
                label = kmedoids(n_group, self.n_user, embedding)

            # Balanced Sparse k-medoids
            elif cluster_var == 'bskm':
                label = kmedoids(n_group, self.n_user, embedding, True)
                
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


    # def randomWalk(self, max_iter=5, featureK=7, var='mean'):
    #     if len(self.rw) == 0:
    #         self.rw = np.zeros((self.n_user, max_iter, featureK))
    #         for i in range(self.n_user):
    #             for it in range(max_iter):
    #                 if len(np.where(self.ind[i] == 1)[0]) == 0:
    #                     self.rw[i, it] = np.array([i] * featureK)
    #                 else:
    #                     init_node = i
    #                     for depth in range(featureK):
    #                         self.rw[i, it, depth] = init_node
    #                         ind = self.ind.copy() if depth % 2 == 0 else self.ind.T.copy()
    #                         rating = self.rating.copy() if depth % 2 == 0 else self.rating.T.copy()
    #                         neighbor = np.where(ind[init_node] == 1)[0]
    #                         weight = rating[init_node, neighbor]
    #                         init_node = np.random.choice(neighbor, p=(weight/weight.sum()))
        
    #     # shape of rw: [userNum, max_iter, featureK]
    #     if var == 'mean':
    #         return np.mean(self.rw, axis=1)
    #     elif var == 'all':
    #         return self.rw.reshape((-1, featureK))


    def PCAClustering(self, n_group, embedding):
        component = PCA(n_components=1).fit_transform(embedding).squeeze()
        sort_idx = np.argsort(component)
        label = np.empty(self.n_user, dtype=int)
        groupLen = self.n_user // n_group
        for i in range(n_group):
            label[sort_idx[i*groupLen : (i+1)*groupLen]] = i
        return label





