import numpy as np
from scipy.sparse import coo_matrix

from method.utils import computeRMSE
from method.utils import computeRMSE2


class InverseSGD(object):
    def __init__(self, param, name='inverseSGD'):
        self.userNum = param['userNum']
        self.itemNum = param['itemNum']
        self.featureK = param['featureK']
        self.epochs = param['epochs']
        self.lambdaU = param['lambdaU']
        self.lambdaV = param['lambdaV']
        self.alpha = param['alpha']
        self.lrDecay = param['lrDecay']
        self.bound = param['bound']

        self.name = name
        self.userMat = []
        self.itemMat = []

        self.loss = []


    def unlearn(self, train_rating, train_rating_ind, del_rating, del_rating_ind, userMat, itemMat, del_user=[], del_point=[], var=2):
        '''
         variance of inverse SGD
        0: naive
        1: equational
        2: equational + early stop
        '''
        assert var in [0, 1, 2]
        self.userMat = userMat.copy()
        self.itemMat = itemMat.copy()

        for id in del_user:
            tmpLoss = 1000
            for epoch in range(self.epochs):
                # learning rate inverse decay
                alpha_dec = self.alpha * (self.lrDecay ** ((self.epochs - epoch) // 50))

                pred = self.userMat @ self.itemMat.T # pred = U * V'
                index_err = train_rating_ind[id-1] * (pred[id-1] - train_rating[id-1]) # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err @ self.itemMat + self.lambdaU * self.userMat[id-1] # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err.reshape(-1, 1) @ self.userMat[id-1].reshape(1, -1) + self.lambdaV * self.itemMat # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # inverse gradient descent
                    self.userMat[id-1] += alpha_dec * deltaU
                    self.itemMat += alpha_dec * deltaV
                else:
                    if var == 2:
                        loss = 0.5 * (computeRMSE(del_rating, del_rating_ind, self.userMat, self.itemMat) ** 2 + \
                            self.lambdaU * np.sum(self.userMat**2) + self.lambdaV * np.sum(self.itemMat**2))
                        self.loss.append(loss)
                        if loss < tmpLoss:
                            tmpLoss = loss
                        else:
                            print('Early stop at epoch: %d' % epoch)
                            break
                    U1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * np.linalg.pinv(index_err.reshape(1, -1)) - alpha_dec * alpha_dec * index_err.reshape(-1, 1)
                    U2 = alpha_dec * self.itemMat + (1 - alpha_dec * self.lambdaV) * np.dot(np.linalg.pinv(index_err.reshape(1, -1)), self.userMat[id - 1].reshape(1, -1))
                    V1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * np.linalg.pinv(index_err.reshape(-1, 1)) - alpha_dec * alpha_dec * index_err.reshape(1, -1)
                    V2 = alpha_dec * self.userMat[id - 1].reshape(1, -1) + (1 - alpha_dec * self.lambdaU) * np.dot(np.linalg.pinv(index_err.reshape(-1, 1)), self.itemMat)

                    self.userMat[id-1] = np.linalg.pinv(U1) @ U2
                    self.itemMat = np.linalg.pinv(V1) @ V2

        for rating in del_point:
            tmpLoss = 1000
            rec = rating.strip().split(' ')
            u = int(rec[0] - 1)
            v = int(rec[1] - 1)
            r = float(rec[2])
            for epoch in range(self.epochs):
                # learning rate inverse decay
                alpha_dec = self.alpha * (self.lrDecay ** ((self.epochs - epoch) // 50))

                pred = userMat @ itemMat.T # pred = U * V'
                index_err = pred[u][v] - r # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err * self.itemMat[v] + self.lambdaU * self.userMat[u] # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err * self.userMat[u] + self.lambdaV * self.itemMat[v] # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # inverse gradient descent
                    self.userMat[u] += alpha_dec * deltaU
                    self.itemMat[v] += alpha_dec * deltaV
                else:
                    if var == 2:
                        loss = 0.5 * (computeRMSE(del_rating, del_rating_ind, self.userMat, self.itemMat) ** 2 + \
                            self.lambdaU * np.sum(self.userMat**2) + self.lambdaV * np.sum(self.itemMat**2))
                        self.loss.append(loss)
                        if loss < tmpLoss:
                            tmpLoss = loss
                        else:
                            print('Early stop at epoch: %d' % epoch)
                            break
                    U1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * index_err - alpha_dec * alpha_dec * index_err
                    U2 = alpha_dec * self.itemMat[v] + (1 - alpha_dec * self.lambdaV) * index_err * self.userMat[u]
                    # V1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * index_err - alpha_dec * alpha_dec * index_err
                    V2 = alpha_dec * self.userMat[u] + (1 - alpha_dec * self.lambdaU) * index_err * self.itemMat[v]

                    self.userMat[u] = U2 / U1
                    self.itemMat[v] = V2 / U1

        return self.userMat, self.itemMat


    def batchUnlearn(self, train_rating, del_rating, userMat, itemMat, batch_size=4096000, var=2):
        '''
         variance of inverse SGD
        0: naive
        1: equational
        2: equational + early stop
        '''
        assert var in [0, 1, 2]
        self.userMat = userMat.copy()
        self.itemMat = itemMat.copy()

        tmpLoss = 1000
        # unlearned rating
        un_rating = list(set(train_rating) - set(del_rating))
        for epoch in range(self.epochs):
            # learning rate inverse decay
            alpha_dec = self.alpha * (self.lrDecay ** ((self.epochs - epoch) // 50))

            cur_index = 0
            np.random.shuffle(un_rating)
            ratingNum = len(un_rating)

            while cur_index < ratingNum:
                cur_batch_rating = un_rating[cur_index : cur_index+batch_size]
                cur_index += batch_size

                cur_user, cur_item, rating, ind = self.__buildMatFromBatch(cur_batch_rating)
                cur_userMat = userMat[cur_user]
                cur_itemMat = itemMat[cur_item]

                pred = cur_userMat @ cur_itemMat.T # pred = U * V'
                index_err = ind * (pred - rating) # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err @ cur_itemMat + self.lambdaU * cur_userMat # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err.T @ cur_userMat + self.lambdaV * cur_itemMat # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # inverse gradient descent
                    cur_userMat += alpha_dec * deltaU
                    cur_itemMat += alpha_dec * deltaV
                else:
                    U1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * np.linalg.pinv(index_err) - alpha_dec * alpha_dec * index_err.T
                    U2 = alpha_dec * cur_itemMat + (1 - alpha_dec * self.lambdaV) * np.dot(np.linalg.pinv(index_err), cur_userMat)
                    V1 = (1 - alpha_dec * self.lambdaU) * (1 - alpha_dec * self.lambdaV) * np.linalg.pinv(index_err).T - alpha_dec * alpha_dec * index_err
                    V2 = alpha_dec * cur_userMat + (1 - alpha_dec * self.lambdaU) * np.dot(np.linalg.pinv(index_err).T, cur_itemMat)

                    cur_userMat = np.linalg.pinv(U1) @ U2
                    cur_itemMat = np.linalg.pinv(V1) @ V2
                # update original mat
                for i in range(len(cur_user)):
                    self.userMat[cur_user[i]] = cur_userMat[i]
                for i in range(len(cur_item)):
                    self.itemMat[cur_item[i]] = cur_itemMat[i]
            if var == 2:
                loss = 0.5 * (computeRMSE2(del_rating, self.userMat, self.itemMat) ** 2 + \
                    self.lambdaU * np.sum(self.userMat**2) + self.lambdaV * np.sum(self.itemMat**2))
                self.loss.append(loss)
                if loss < tmpLoss:
                    tmpLoss = loss
                else:
                    print('Early stop at epoch: %d' % epoch)
                    break

        return self.userMat, self.itemMat

    def __buildMatFromBatch(self, cur_batch_rating):
        cur_user = []
        cur_item = []
        row, col, val, ind = [], [], [], []
        for line in cur_batch_rating:
            cur_line = line.strip().split(',')
            user = int(cur_line[0])
            item = int(cur_line[1])
            rating = float(cur_line[2])
            if user not in cur_user:
                cur_user.append(user)
            if item not in cur_item:
                cur_item.append(item)
            row.append(cur_user.index(user))
            col.append(cur_item.index(item))
            val.append(rating)
            ind.append(1)
        cur_userNum = len(cur_user)
        cur_itemNum = len(cur_item)
        val_matrix = coo_matrix((val, (row, col)), shape=(cur_userNum, cur_itemNum))
        ind_matrix = coo_matrix((ind, (row, col)), shape=(cur_userNum, cur_itemNum))
        return cur_user, cur_item, val_matrix.toarray(), ind_matrix.toarray()
