import _numpy as np
from scipy.sparse import coo__matrix

from method.utils import computeRMSE
from method.utils import computeRMSE2


# class ReverseHessian(object):
#     def __init__(self, param, name='reverseHessian'):
#         self.name = name
#         self.user_mat = []
#         self.item_mat = []

    
#     def unlearn(self):
#         self.user_mat = userBar + 
#         self.item_mat = itemBar + 
#         return self.user_mat, self.item_mat

class ReverseSGD(object):
    def __init__(self, param, name='reverseSGD'):
        self.user_num = param['user_num']
        self.item_num = param['item_num']
        self.k = param['k']
        self.epochs = param['epochs']
        self.lambda_u = param['lambda_u']
        self.lambda_v = param['lambda_v']
        self.lr = param['lr']
        self.lr_decay = param['lr_decay']
        self.bound = param['bound']

        self.name = name
        self.user_mat = []
        self.item_mat = []

        self.loss = []


    def unlearn(self, train_rating, train_rating_ind, del_rating, del_rating_ind, user_mat, item_mat, del_user=[], del_point=[], var=2):
        '''
         variance of reverse SGD
        0: naive
        1: equational
        2: equational + early stop
        '''
        assert var in [0, 1, 2]
        self.user_mat = user_mat.copy()
        self.item_mat = item_mat.copy()

        for id in del_user:
            tmpLoss = 1000
            for epoch in range(self.epochs):
                # learning rate reverse decay
                lr_dec = self.lr * (self.lr_decay ** ((self.epochs - epoch) // 50))

                pred = self.user_mat @ self.item_mat.T # pred = U * V'
                index_err = train_rating_ind[id-1] * (pred[id-1] - train_rating[id-1]) # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err @ self.item_mat + self.lambda_u * self.user_mat[id-1] # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err.reshape(-1, 1) @ self.user_mat[id-1].reshape(1, -1) + self.lambda_v * self.item_mat # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # reverse gradient descent
                    self.user_mat[id-1] += lr_dec * deltaU
                    self.item_mat += lr_dec * deltaV
                else:
                    if var == 2:
                        loss = 0.5 * (computeRMSE(del_rating, del_rating_ind, self.user_mat, self.item_mat) ** 2 + \
                            self.lambda_u * np.sum(self.user_mat**2) + self.lambda_v * np.sum(self.item_mat**2))
                        self.loss.append(loss)
                        if loss < tmpLoss:
                            tmpLoss = loss
                        else:
                            print('Early stop at epoch: %d' % epoch)
                            break
                    U1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * np.linalg.pinv(index_err.reshape(1, -1)) - lr_dec * lr_dec * index_err.reshape(-1, 1)
                    U2 = lr_dec * self.item_mat + (1 - lr_dec * self.lambda_v) * np.dot(np.linalg.pinv(index_err.reshape(1, -1)), self.user_mat[id - 1].reshape(1, -1))
                    V1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * np.linalg.pinv(index_err.reshape(-1, 1)) - lr_dec * lr_dec * index_err.reshape(1, -1)
                    V2 = lr_dec * self.user_mat[id - 1].reshape(1, -1) + (1 - lr_dec * self.lambda_u) * np.dot(np.linalg.pinv(index_err.reshape(-1, 1)), self.item_mat)

                    self.user_mat[id-1] = np.linalg.pinv(U1) @ U2
                    self.item_mat = np.linalg.pinv(V1) @ V2

        for rating in del_point:
            tmpLoss = 1000
            rec = rating.strip().split(' ')
            u = int(rec[0] - 1)
            v = int(rec[1] - 1)
            r = float(rec[2])
            for epoch in range(self.epochs):
                # learning rate reverse decay
                lr_dec = self.lr * (self.lr_decay ** ((self.epochs - epoch) // 50))

                pred = user_mat @ item_mat.T # pred = U * V'
                index_err = pred[u][v] - r # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err * self.item_mat[v] + self.lambda_u * self.user_mat[u] # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err * self.user_mat[u] + self.lambda_v * self.item_mat[v] # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # reverse gradient descent
                    self.user_mat[u] += lr_dec * deltaU
                    self.item_mat[v] += lr_dec * deltaV
                else:
                    if var == 2:
                        loss = 0.5 * (computeRMSE(del_rating, del_rating_ind, self.user_mat, self.item_mat) ** 2 + \
                            self.lambda_u * np.sum(self.user_mat**2) + self.lambda_v * np.sum(self.item_mat**2))
                        self.loss.append(loss)
                        if loss < tmpLoss:
                            tmpLoss = loss
                        else:
                            print('Early stop at epoch: %d' % epoch)
                            break
                    U1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * index_err - lr_dec * lr_dec * index_err
                    U2 = lr_dec * self.item_mat[v] + (1 - lr_dec * self.lambda_v) * index_err * self.user_mat[u]
                    # V1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * index_err - lr_dec * lr_dec * index_err
                    V2 = lr_dec * self.user_mat[u] + (1 - lr_dec * self.lambda_u) * index_err * self.item_mat[v]

                    self.user_mat[u] = U2 / U1
                    self.item_mat[v] = V2 / U1

        return self.user_mat, self.item_mat


    def batchUnlearn(self, train_rating, del_rating, user_mat, item_mat, batch_size=4096000, var=2):
        '''
         variance of reverse SGD
        0: naive
        1: equational
        2: equational + early stop
        '''
        assert var in [0, 1, 2]
        self.user_mat = user_mat.copy()
        self.item_mat = item_mat.copy()

        tmpLoss = 1000
        # unlearned rating
        un_rating = list(set(train_rating) - set(del_rating))
        for epoch in range(self.epochs):
            # learning rate reverse decay
            lr_dec = self.lr * (self.lr_decay ** ((self.epochs - epoch) // 50))

            cur_index = 0
            np.random.shuffle(un_rating)
            rating_num = len(un_rating)

            while cur_index < rating_num:
                cur_batch_rating = un_rating[cur_index : cur_index+batch_size]
                cur_index += batch_size

                cur_user, cur_item, rating, ind = self.__build_matFromBatch(cur_batch_rating)
                cur_user_mat = user_mat[cur_user]
                cur_item_mat = item_mat[cur_item]

                pred = cur_user_mat @ cur_item_mat.T # pred = U * V'
                index_err = ind * (pred - rating) # index_err = I * (U * V' - R)
                if var == 0:
                    deltaU = index_err @ cur_item_mat + self.lambda_u * cur_user_mat # delta_U = I * (U * V' - R) * V + lambdas_U * U
                    deltaV = index_err.T @ cur_user_mat + self.lambda_v * cur_item_mat # delta_V = I * (U * V' - R)' * U + lambdas_V * V
                    # reverse gradient descent
                    cur_user_mat += lr_dec * deltaU
                    cur_item_mat += lr_dec * deltaV
                else:
                    U1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * np.linalg.pinv(index_err) - lr_dec * lr_dec * index_err.T
                    U2 = lr_dec * cur_item_mat + (1 - lr_dec * self.lambda_v) * np.dot(np.linalg.pinv(index_err), cur_user_mat)
                    V1 = (1 - lr_dec * self.lambda_u) * (1 - lr_dec * self.lambda_v) * np.linalg.pinv(index_err).T - lr_dec * lr_dec * index_err
                    V2 = lr_dec * cur_user_mat + (1 - lr_dec * self.lambda_u) * np.dot(np.linalg.pinv(index_err).T, cur_item_mat)

                    cur_user_mat = np.linalg.pinv(U1) @ U2
                    cur_item_mat = np.linalg.pinv(V1) @ V2
                # update original _mat
                for i in range(len(cur_user)):
                    self.user_mat[cur_user[i]] = cur_user_mat[i]
                for i in range(len(cur_item)):
                    self.item_mat[cur_item[i]] = cur_item_mat[i]
            if var == 2:
                loss = 0.5 * (computeRMSE2(del_rating, self.user_mat, self.item_mat) ** 2 + \
                    self.lambda_u * np.sum(self.user_mat**2) + self.lambda_v * np.sum(self.item_mat**2))
                self.loss.append(loss)
                if loss < tmpLoss:
                    tmpLoss = loss
                else:
                    print('Early stop at epoch: %d' % epoch)
                    break

        return self.user_mat, self.item_mat

    def __build_matFromBatch(self, cur_batch_rating):
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
        cur_user_num = len(cur_user)
        cur_item_num = len(cur_item)
        val__matrix = coo__matrix((val, (row, col)), shape=(cur_user_num, cur_item_num))
        ind__matrix = coo__matrix((ind, (row, col)), shape=(cur_user_num, cur_item_num))
        return cur_user, cur_item, val__matrix.toarray(), ind__matrix.toarray()
