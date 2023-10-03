import numpy as np
from itertools import combinations
from numpy.random import triangular
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


class Group(object):
    def __init__(self, rating, ind, userMat=None):
        self.rating = rating
        self.ind = ind
        self.userMat = userMat
        self.userNum = self.rating.shape[0]
        self.itemNum = self.rating.shape[1]

        # build UI graph (adjacency matrix)
        # left = np.vstack((np.zeros((self.userNum, self.userNum), dtype=np.float16), self.rating.T))
        # right = np.vstack((self.rating, np.zeros((self.itemNum, self.itemNum), dtype=np.float16)))
        # self.ui = np.hstack((left, right))

        self.user = []
        self.rw = []
        self.nm = []

    
    def UIGrouping(self, groupNum=2, var='spectral', verbose=True):
        assert groupNum > 1

        if var == 'spectral':
            cluster = SpectralClustering(n_clusters=groupNum, 
                                            affinity='precomputed',
                                            assign_labels='discretize').fit(self.ui)
            label = cluster.labels_[:self.userNum]
        
        if verbose == True:
            # print result
            result = ''
            for i in range(groupNum):
                num = str(i) + ': ' + str(len(np.where(label == i)[0])) + ', '
                result += num
            print(result)
        return label


    def UI2U(self, var='pearson'):
        # transform UI matrix to User adjacency matrix
        user_aj = np.zeros((self.userNum, self.userNum), dtype=np.float16)

        # if [i, j] have rating on same item, aj[i, j] += 1
        if var == 'bool':
            for ind in self.ind.T:
                have_rating = np.where(ind == 1)[0]
                for comb in combinations(have_rating, 2):
                    user_aj[comb] += 1
            user_aj += user_aj.T

        # aj[i, j] += (max - |rating_i - rating_j|)
        elif var == 'float':
            max_rating = 1
            combs = combinations(range(self.userNum), 2)
            for rating in self.rating.T:
                for comb in combs:
                    user_aj[comb] += max_rating - np.abs(rating[comb[0]] - rating[comb[1]])
            user_aj += user_aj.T

        # regard user's rating on all items as embedding, compute EU dist
        elif var == 'euclidean':
            user_aj = -euclidean_distances(self.rating)

        # compute cosine dist
        elif var == 'cosine':
            user_aj = -cosine_distances(self.rating)

        # compute manhattan dist
        elif var == 'manhattan':
            user_aj = -manhattan_distances(self.rating)
        
        # pearson similarity
        elif var == 'pearson':
            combs = combinations(range(self.userNum), 2)
            for comb in combs:
                # intersection - implement 1
                # have_rating1 = np.where(self.ind[comb[0]] == 1)[0]
                # have_rating2 = np.where(self.ind[comb[1]] == 1)[0]
                # common_rating = list(set(have_rating1).intersection(set(have_rating2)))
                # intersection - implement 2
                # common_ind = self.ind[comb[0]] * self.ind[comb[1]]
                # common_rating = np.where(common_ind == 1)[0]
                # if len(common_rating) < 2:
                #     similarity = 0
                # else:
                #     # intersection
                #     rating1 = self.rating[comb[0]][common_rating]
                #     rating2 = self.rating[comb[1]][common_rating]
                #     similarity = stats.pearsonr(rating1, rating2)[0]
                # all items
                if len(np.where(self.ind[comb[0]] == 1)[0]) == 0 or len(np.where(self.ind[comb[1]] == 1)[0]) == 0:
                    similarity = 0
                else:
                    rating1 = self.rating[comb[0]]
                    rating2 = self.rating[comb[1]]
                    similarity = stats.pearsonr(rating1, rating2)[0]
                user_aj[comb] = similarity
            user_aj += user_aj.T
        # self.user = user_aj
        return user_aj


    def UserGrouping(self, groupNum=2, var='spectral-euclidean', verbose=True):
        assert groupNum > 1
        [cluster_var, trans_var] = var.strip().split('-')
        # graph-based cluster / community detection
        if cluster_var in ['spectral', 'lpa', 'blpa']:
            assert trans_var in ['bool', 'float', 'euclidean', 'pearson', 'cosine', 'manhattan']
            self.user = self.UI2U(trans_var)
            # spectral clustering
            if cluster_var == 'spectral':
                cluster = SpectralClustering(n_clusters=groupNum, 
                                                affinity='precomputed',
                                                assign_labels='discretize').fit(self.user)
                label = cluster.labels_
            # Label Propagation Algorithm (LPA)
            elif cluster_var == 'lpa':
                label = self.lpa(groupNum)
            # Balanced LPA
            elif cluster_var == 'blpa':
                label = self.lpa(groupNum, True)
        
        # embedding + clustering
        elif cluster_var in ['kmeans', 'bkmeans', 'bekm', 'pca']:
            #### embedding ####
            if trans_var == 'rating':
                embedding = self.rating
            elif trans_var == 'mf':
                embedding = self.userMat
            elif trans_var == 'neighbor_mean':
                if len(self.nm) == 0:
                    embedding = np.zeros((self.userNum, self.userNum), dtype=np.float16)
                    for i in range(self.userNum):
                        have_rating = np.where(self.ind[i] == 1)[0]
                        if len(have_rating) > 0:
                            embedding[i] = np.mean(self.rating.T[have_rating], axis=0)
                    self.nm = embedding
                else:
                    embedding = self.nm
            elif trans_var == 'random_walk':
                embedding = self.randomWalk()
            elif trans_var == 'deep_walk':
                embedding = self.randomWalk(var='all')

            #### clustering ####
            # k-means
            if cluster_var in ['kmeans', 'bkmeans']:
                # cluster_method = eval('KMeans') if cluster_var == 'kmeans' else eval('MiniBatchKMeans')
                # cluster = cluster_method(n_clusters=groupNum).fit(embedding)
                # label = cluster.labels_
                if cluster_var == 'kmeans':
                    label = self.kmeans(groupNum, embedding)
                elif cluster_var == 'bkmeans':
                    cluster = MiniBatchKMeans(n_clusters=groupNum).fit(embedding)
                    label = cluster.labels_
            # Balanced Embedding k-mean (BEKM)
            elif cluster_var == 'bekm':
                label = self.kmeans(groupNum, embedding, True)
            # PCA based clustering
            elif cluster_var == 'pca':
                label = self.PCAClustering(groupNum, embedding)
        
        if verbose == True:
            # print result
            result = ''
            for i in range(groupNum):
                num = str(i) + ': ' + str(len(np.where(label == i)[0])) + ', '
                result += num
            print(result)
        return label


    def singleLPA(self, groupNum, balanced, max_iter):
        groupLen = 1 + self.userNum // groupNum
        # init
        label = np.random.randint(0, groupNum, size=(self.userNum))

        for i in range(max_iter):
            group_weight = np.zeros((self.userNum, groupNum))
            for user in range(self.userNum):
                group_weight[:, label[user]] += np.exp(self.user[user]) # similarity to weight

            # LPA
            if balanced == False:
                new_label = group_weight.argmax(axis=1)
            # BPLA
            else:
                new_label = np.zeros_like(label)
                label_count = [groupLen] * groupNum
                sim_list = self.buildDict(group_weight)[::-1]
                assigned = []
                for j in range(len(sim_list)):
                    user_idx = sim_list[j][0][0]
                    if user_idx in assigned:
                        continue
                    cluster_idx = sim_list[j][0][1]
                    if label_count[cluster_idx] > 0:
                        new_label[user_idx] = cluster_idx
                        assigned.append(user_idx)
                        label_count[cluster_idx] -= 1
            inertia = np.sum(group_weight[np.arange(self.userNum), new_label])

            if (new_label == label).all():
                break
            label = new_label
        return label, inertia


    def lpa(self, groupNum, balanced=False, n_init=10, max_iter=30):
        # np.random.seed(0)
        tmp_inertia = 1e10
        for i in range(n_init):
            label, inertia = self.singleLPA(groupNum, balanced, max_iter)
            if inertia < tmp_inertia:
                tmp_inertia = inertia
                fin_label = label
        return fin_label


    def randomWalk(self, max_iter=5, featureK=7, var='mean'):
        if len(self.rw) == 0:
            self.rw = np.zeros((self.userNum, max_iter, featureK))
            for i in range(self.userNum):
                for it in range(max_iter):
                    if len(np.where(self.ind[i] == 1)[0]) == 0:
                        self.rw[i, it] = np.array([i] * featureK)
                    else:
                        init_node = i
                        for depth in range(featureK):
                            self.rw[i, it, depth] = init_node
                            ind = self.ind.copy() if depth % 2 == 0 else self.ind.T.copy()
                            rating = self.rating.copy() if depth % 2 == 0 else self.rating.T.copy()
                            neighbor = np.where(ind[init_node] == 1)[0]
                            weight = rating[init_node, neighbor]
                            init_node = np.random.choice(neighbor, p=(weight/weight.sum()))
        
        # shape of rw: [userNum, max_iter, featureK]
        if var == 'mean':
            return np.mean(self.rw, axis=1)
        elif var == 'all':
            return self.rw.reshape((-1, featureK))


    def buildDict(self, sim):
        sim_dict = {}
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                sim_dict[i, j] = sim[i, j]
        return sorted(sim_dict.items(), key=lambda d:(d[1], d[0])) # [((i, j), sim), ...]


    def singleKmeans(self, k, embedding, balanced, max_iter):
        label = np.zeros(self.userNum, dtype=int)
        groupLen = 1 + self.userNum // k
        # init
        cen_idx = np.random.choice(self.userNum, k, replace=False)
        centroid = embedding[cen_idx].copy()

        e_square = np.expand_dims(np.sum(embedding * embedding, axis=1), 1)
        for i in range(max_iter):
            dist = -2 * np.dot(embedding, centroid.T)
            dist += e_square
            dist += np.expand_dims(np.sum(centroid * centroid, axis=1), 0)

            # kmeans
            if balanced == False:
                new_label = dist.argmin(axis=1)
            # balanced kmeans
            else:
                new_label = np.zeros_like(label)
                label_count = [groupLen] * k
                dist_list = self.buildDict(dist)
                assigned = []
                for j in range(len(dist_list)):
                    user_idx = dist_list[j][0][0]
                    if user_idx in assigned:
                        continue
                    cluster_idx = dist_list[j][0][1]
                    if label_count[cluster_idx] > 0:
                        new_label[user_idx] = cluster_idx
                        assigned.append(user_idx)
                        label_count[cluster_idx] -= 1
            inertia = np.sum(dist[np.arange(self.userNum), new_label])

            if (new_label == label).all():
                break
            label = new_label
            for j in range(k):
                centroid[j] = np.mean(embedding[label == j], axis=0)
        return label, inertia


    def kmeans(self, groupNum, embedding, balanced=False, n_init=10, max_iter=300):
        # np.random.seed(0)
        tmp_inertia = 1e10
        for i in range(n_init):
            label, inertia = self.singleKmeans(groupNum, embedding, balanced, max_iter)
            if inertia < tmp_inertia:
                tmp_inertia = inertia
                fin_label = label
        return fin_label


    def PCAClustering(self, groupNum, embedding):
        component = PCA(n_components=1).fit_transform(embedding).squeeze()
        sort_idx = np.argsort(component)
        label = np.empty(self.userNum, dtype=int)
        groupLen = self.userNum // groupNum
        for i in range(groupNum):
            if i < groupNum - 1:
                label[sort_idx[i*groupLen : (i+1)*groupLen]] = i
            else:
                label[sort_idx[i*groupLen : self.userNum]] = i
        return label





