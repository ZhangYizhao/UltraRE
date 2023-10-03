from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from sklearn.metrics.pairwise import pairwise_distances
import warnings

import torch
from torch import nn
import ot

##################### 
# model training
##################### 

# seed everything
def seed_all(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

STD = 1 # original code: 0.01
# build model MF
class MF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(MF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(1)


# build model Generalized MF
class GMF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(GMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)

        self.affine = nn.Linear(self.k, 1)
        self.logistic = nn.Sigmoid()
        
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        nn.init.xavier_uniform_(self.affine.weight)

    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        logits = self.affine(torch.mul(user_embedding, item_embedding))
        rating = self.logistic(logits)
        return rating.squeeze()


# build model DMF
class DMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layers=[64, 32]):
        super(DMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.layers = [k]
        self.layers += layers
        self.user_fc = nn.ModuleList()
        self.item_fc = nn.ModuleList()
        self.cos = nn.CosineSimilarity()

        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.user_fc.append(nn.Linear(in_size, out_size))
            self.item_fc.append(nn.Linear(in_size, out_size))
            self.user_fc.append(nn.ReLU())
            self.item_fc.append(nn.ReLU())

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()


    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        for i in range(len(self.user_fc)):
            user_embedding = self.user_fc[i](user_embedding)
            item_embedding = self.item_fc[i](item_embedding)
        rating = self.cos(user_embedding, item_embedding)
        return rating.squeeze()


# build model Neural MF
class NMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layser=[64, 32]):
        super(NMF, self).__init__()
        self.k = k
        self.k_mlp = int(layser[0]/2)

        self.user_mat_mf = nn.Embedding(n_user, k)
        self.item_mat_mf = nn.Embedding(n_item, k)
        self.user_mat_mlp = nn.Embedding(n_user, self.k_mlp)
        self.item_mat_mlp = nn.Embedding(n_item, self.k_mlp)

        self.layers = layser
        self.fc = nn.ModuleList()
        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.fc.append(nn.Linear(in_size, out_size))
            self.fc.append(nn.ReLU())

        self.affine = nn.Linear(self.layers[-1] + self.k, 1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat_mf.weight, std=STD)
        nn.init.normal_(self.item_mat_mf.weight, std=STD)
        nn.init.normal_(self.user_mat_mlp.weight, std=STD)
        nn.init.normal_(self.item_mat_mlp.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()
        # for i in self.fc:
        #     if isinstance(i, nn.Linear):
        #         nn.init.xavier_uniform_(i.weight)
        #         if i.bias is not None:
        #             i.bias.data.zero_()

        # nn.init.xavier_uniform_(self.affine.weight)
        # if self.affine.bias is not None:
        #     self.affine.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding_mlp = self.user_mat_mlp(uid)
        item_embedding_mlp = self.item_mat_mlp(iid)

        user_embedding_mf = self.user_mat_mf(uid)
        item_embedding_mf = self.item_mat_mf(iid)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vec = torch.mul(user_embedding_mf,item_embedding_mf)

        for i in range(len(self.fc)):
            mlp_vec = self.fc[i](mlp_vec)

        vec = torch.cat([mlp_vec, mf_vec], dim=-1)
        logits = self.affine(vec)
        rating = self.logistic(logits)
        return rating.squeeze()


# torch train
def baseTrain(dataloader, model, loss_fn, is_rmse, opt, device, verbose, var='nor', attr=[]):
    assert var in ['nor', 'u2u', 'd2d']
    if var != 'nor':
        id1, id2 = attr
        id1 = torch.tensor(id1, device=device)
        id2 = torch.tensor(id2, device=device)
    size = len(dataloader.dataset)
    train_loss = 0
    rmse_loss = 0
    if is_rmse == False:
        rmse_fn = nn.MSELoss(reduction='sum')

    for batch, (user, item, rating) in enumerate(dataloader):
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)

        pred = model(user, item)
        if var == 'nor':
            loss = loss_fn(pred, rating)
        else:
            # divide user by attribute
            uni_user = user.unique()
            user1 = uni_user[torch.isin(uni_user, id1)]
            user2 = uni_user[torch.isin(uni_user, id2)]
            user_mat1 = model.user_mat_mf(user1)  # to be modified when using LightGCN
            user_mat2 = model.user_mat_mf(user2)  # to be modified when using LightGCN
            if var == 'u2u':
                user_mat = model.user_mat_mf.weight  # to be modified when using LightGCN
                lap_mat = buildLap(user_mat.size(0), user1.cpu(), user2.cpu())
                eta = 1
                dis_loss = torch.trace(torch.mm(user_mat.T, torch.sparse.mm(lap_mat.to(device), user_mat)))
                loss = loss_fn(pred, rating) + eta * dis_loss
            elif var == 'd2d':
                eta = 1
                loss = loss_fn(pred, rating) + eta * mmd_loss(user_mat1, user_mat2)
        train_loss += loss.item()

        if is_rmse == False:
            with torch.no_grad():
                rmse = rmse_fn(pred, rating)
                rmse_loss += rmse.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if is_rmse == False:
            loss_val = loss.item() / len(user)
            rmse_val = np.sqrt(rmse.item() / len(user))
        else:
            loss_val = np.sqrt(loss.item() / len(user))
            rmse_val = loss_val

        if verbose == 2 and batch % 100 == 0:
            cur_batch = batch * len(user)
            print(f'Train - Loss: {loss_val:>.6f}, RMSE: {rmse_val:>.4f} [{cur_batch:>8d}/{size:>8d}]')

    if is_rmse == False:
        train_loss /= size
        rmse_loss = np.sqrt(rmse_loss / size)
    else:
        train_loss = np.sqrt(train_loss / size)
        rmse_loss = train_loss

    return train_loss, rmse_loss


# torch test
def baseTest(dataloader, models, loss_fn, device, verbose, top_k=10):
    '''
    Parameters
    ----------
    models: list [n_group]
    '''
    size = len(dataloader.dataset)
    for model in models:
        model.eval()
    test_loss, ndcg, hr = 0, [], []
    with torch.no_grad():
        all_user = []
        rating_dict = {}
        for user, item, rating in dataloader:
            # to list or array
            uni_user = user.unique().tolist()
            batch_user = user.numpy().astype(np.int32)
            batch_rating = rating.numpy().astype(np.float32)

            # to GPU device
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            
            # prediction
            preds = []
            for model in models:
                model.to(device)
                tmp_pred = model(user, item)
                preds.append(tmp_pred)
            pred = torch.stack(preds).mean(dim=0).to(device)

            # compute loss
            test_loss += loss_fn(pred, rating).item()

            # compute NDCG & HR
            batch_pred = pred.cpu().numpy().astype(np.float32).reshape(-1)
            for uid in uni_user:
                cur_rating = batch_rating[batch_user == uid].tolist()
                cur_pred = batch_pred[batch_user == uid].tolist()
                if uid in all_user:
                    rating_dict[uid]['rating'] += cur_rating
                    rating_dict[uid]['pred'] += cur_pred
                else:
                    all_user.append(uid)
                    rating_dict[uid] = {'rating': cur_rating,
                                        'pred': cur_pred}
        # compute loss
        test_loss = np.sqrt(test_loss / size)

        # compute NDCG & HR
        for uid in all_user:
            uid_rating = np.array(rating_dict[uid]['rating'])
            uid_pred = np.array(rating_dict[uid]['pred'])

            top_rating = np.argsort(uid_rating)[::-1][:top_k]
            top_pred = np.argsort(uid_pred)[::-1][:top_k]

            # HR
            relevance = uid_rating[top_pred]
            n_hit = sum(relevance >= (4/5))
            hr.append(n_hit/top_k)
            
            # NDCG
            common_idx = np.in1d(top_rating, top_pred)
            relevance *= (relevance >= (4/5))
            ndcg.append(computeNDCG(relevance * common_idx, top_k))

        ndcg = np.mean(ndcg)
        hr = np.mean(hr)
    if verbose == 2:
        print(f'Test - RMSE: {test_loss:>.4f}, NDCG: {ndcg:>.3f}, HR: {hr:>.3f}')
    return test_loss, ndcg, hr

# >>> compute NDCG
def computeNDCG(r, top_k):
    '''
    Parameters
    ----------
    r:      array [n_sample]
    top_k:  int
    
    Returns
    -------
    NDCG:   weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
    '''
    n = len(r)
    if n == 0:
        return 0
    for _ in range(top_k - n):
        r = np.append(r, 0)
    assert len(r) == top_k
    return computeDCG(r) / computeDCG(np.ones(top_k))

def computeDCG(r):
    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, len(r) + 1)))
# <<< compute NDCG

# shrink and perturb
def spTrick(model, shrink=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param.data[i][j] = shrink * param.data[i][j] + torch.normal(0.0, sigma, size=(1, 1))
    return model

# MMD
def rbk(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Params: 
	    source: [n_sample1, n_dim]
	    target: [n_sample2, n_dim]
	    kernel_mul: 
	    kernel_num:
	    fix_sigma:
	Return:
		sum(kernel_val):
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_dim = int(source.size(0))
    kernels = rbk(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:source_dim, :source_dim]
    YY = kernels[source_dim:, source_dim:]
    XY = kernels[:source_dim, source_dim:]
    YX = kernels[source_dim:, :source_dim]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) -torch.mean(YX)
    return loss

# laplacian
def buildLap(n_user, user1, user2):
    row, col, val = [], [], []
    for i in user1:
        for j in user2:
            row += [i, j]
            col += [j, i]
            val += [1, 1]
    adj_mat = coo_matrix((val, (row, col)), shape=(n_user, n_user))
    lap_mat = laplacian(adj_mat.todense())
    return torch.FloatTensor(lap_mat)

##################### 
# mat visualization
##################### 

def embeddingUI(data, userset, featureK):
    # data = [user_mat, item_mat]
    pca = PCA(n_components=2)
    for i in range(2):
        if i == 0:
            user_mat = data[0].copy()
            for j in range(data[0].shape[0]):
                if j not in userset:
                    user_mat[j] = np.zeros(featureK)
        title = 'USER' if i == 0 else 'ITEM'
        pca.fit(data[i])
        feature2d = pca.transform(data[i])
        ax = plt.subplot(int('12'+str(i+1)))
        plt.scatter(feature2d[:,0], feature2d[:,1])
        ax.set_title(title)

def embeddingItem(data, id=[53, 54, 58, 59]):
    # data = item_mat
    pca = PCA(n_components=2)
    res = np.empty((len(id), data.shape[1]))
    for i, idx in enumerate(id):
        res[i] = data[idx-1]
    pca.fit(res)
    feature2d = pca.transform(res)
    plt.figure()
    plt.scatter(feature2d[:,0], feature2d[:,1])
    for i in range(len(id)):
        plt.annotate(id[i], xy=(feature2d[i,0], feature2d[i,1]))


##################### 
# object saving
##################### 

def saveObject(filename, obj):
    with open(filename + '.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObject(filename):
    with open(filename + '.pkl', 'rb') as input:
        obj = pickle.load(input)
    return obj


##################### 
# clustering & community detection
##################### 

# build ascending list
def sortArr(sim, order='asc'):
    '''
    Parameters
    ----------
    sim:        array [n_user, n_group]
    order:      str in ['asc', 'des']
    
    Returns
    -------
    zip(raw, col):  raw - list [n_user]
                    col - list [n_group]
    '''
    if order == 'asc':
        idx = np.argsort(sim, axis=None)
    else:
        idx = np.argsort(sim, axis=None)[::-1]
    raw, col = np.unravel_index(idx, sim.shape)
    return zip(raw, col)


def singleKmeans(k, n_user, sp_mat, balanced, max_iter):
    '''
    Parameters
    ----------
    sp_mat:     csr_matrix [n_user, n_embedding]

    Returns
    -------
    label:      array of int [n_user]
    inertia:    float
    '''
    label = np.zeros(n_user, dtype=int)
    group_len = int(np.ceil(n_user/k))
    # init
    cen_idx = np.random.choice(n_user, k, replace=False)
    centroid = sp_mat[cen_idx].copy()

    e_square = sp_mat.multiply(sp_mat).sum(axis=1)  # dense mat
    for _ in range(max_iter):
        dist = (-2 * sp_mat * centroid.T).A  # dense mat
        dist += e_square
        dist += centroid.multiply(centroid).sum(axis=1).reshape(1, -1)  # dense mat

        # kmeans
        if balanced == False:
            new_label = dist.argmin(axis=1)
        # balanced kmeans
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * k
            dist_zip = sortArr(dist)
            assigned = []
            for (user_idx, group_idx) in dist_zip:
                if len(assigned) == n_user:
                    break
                if user_idx in assigned:
                    continue
                if label_count[group_idx] > 0:
                    new_label[user_idx] = group_idx
                    assigned.append(user_idx)
                    label_count[group_idx] -= 1
        inertia = np.sum(dist[np.arange(n_user), new_label])

        if (new_label == label).all():
            break
        label = new_label
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for j in range(k):
                centroid[j] = csr_matrix(sp_mat[label == j].mean(axis=0))
    return label, inertia

def kmeans(n_group, n_user, sp_mat, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    sp_mat:     csr_matrix [n_user, n_embedding]
    '''
    tmp_inertia = 1e10
    for _ in range(n_init):
        label, inertia = singleKmeans(n_group, n_user, sp_mat, balanced, max_iter)
        if inertia < tmp_inertia:
            tmp_inertia = inertia
            fin_label = label
    return fin_label


# build AJ matrix with n_neighbor
def findNeighbor(cache_dir, sp_mat, n_user, var='euclidean', n_neighbor=10):
    '''
    Parameters
    ----------
    sp_mat:     csr_matrix [n_user, n_embedding]

    Returns
    -------
    nei_idx:    2d-array [n_user, n_neighbor] index
    nei_val:    2d-array [n_user, n_neighbor] value
    '''
    if cache_dir == True:
        cache = np.load(cache_dir + var + '.npy')
        nei_idx = cache[0]
        nei_val = cache[1]
    else:
        nei_idx = np.zeros((n_user, n_neighbor), dtype=int)
        nei_val = np.zeros((n_user, n_neighbor), dtype=np.float16)
        for i in range(n_user):
            if var == 'euclidean':
                dist = pairwise_distances(sp_mat, sp_mat[i], metric='euclidean', n_jobs=-1)
            elif var == 'cosine':
                dist = pairwise_distances(sp_mat, sp_mat[i], metric='cosine', n_jobs=-1)
            elif var == 'manhattan':
                dist = pairwise_distances(sp_mat, sp_mat[i], metric='manhattan', n_jobs=-1)
            top_idx = np.argsort(dist.squeeze())[:n_neighbor]
            top_val = -1 * np.sort(dist.squeeze())[:n_neighbor]
            nei_idx[i] = top_idx
            nei_val[i] = top_val

        # save
        np.save(cache_dir + var, [nei_idx, nei_val])

    return nei_idx, nei_val


def singleLPA(n_group, n_user, dist_arr, balanced, n_neighbor, max_iter=10):
    '''
    Parameters
    ----------
    dist_arr:   2d-array [n_user, n_user]
    '''
    group_len = int(np.ceil(n_user/n_group))
    # init
    label = np.random.randint(0, n_group, size=(n_user))
    # aj_idx = np.argsort(dist_arr)[:, :n_neighbor]
    # aj_val = -1 * np.sort(dist_arr)[:, :n_neighbor]

    for i in range(max_iter):
        print('iter', i)
        # similarity to weight
        group_weight = np.zeros((n_user, n_group))
        for user_i in range(n_user):
            group_weight[:, label[user_i]] += np.exp(-dist_arr[user_i])
            # for nei_i in range(n_neighbor):
            #     group_idx = label[aj_idx[user_i, nei_i]]
            #     weight = np.exp(aj_val[user_i, nei_i])
            #     group_weight[user_i, group_idx] += weight

        # LPA
        if balanced == False:
            new_label = group_weight.argmax(axis=1)
        # BPLA
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * n_group
            sim_zip = sortArr(group_weight, 'des')
            assigned = []
            for (user_idx, group_idx) in sim_zip:
                if len(assigned) == n_user:
                    break
                if user_idx in assigned:
                    continue
                if label_count[group_idx] > 0:
                    new_label[user_idx] = group_idx
                    assigned.append(user_idx)
                    label_count[group_idx] -= 1
        inertia = np.sum(group_weight[np.arange(n_user), new_label])

        if (new_label == label).all():
            break
        label = new_label
    return label, inertia

def lpa(n_group, n_user, dist_arr, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    dist_arr:   2d-array [n_user, n_user]
    '''
    tmp_inertia = 1e10
    for i in range(n_init):
        print('init', i, '-------')
        label, inertia = singleLPA(n_group, n_user, dist_arr, n_user, balanced, max_iter)
        if inertia < tmp_inertia:
            tmp_inertia = inertia
            fin_label = label
    return fin_label


def genDistArray(user_idx, cen_idx, dist_dict):
    '''
    Parameters
    ----------
    cen_idx:    array [n_cen]
    dist_dict:  dict {(i, j): dist}

    Returns
    -------
    dist:      array [n_user, n_cen]
    '''
    dist = np.zeros((len(user_idx), len(cen_idx)), dtype=np.float32)

    for i in user_idx:
        for j in cen_idx:
            if i == j:
                val = 0
            elif (i, j) in dist_dict:
                val = dist_dict[i, j]
            else:
                val = dist_dict[j, i]
            dist[i, j] = val
    return dist

def singleKmedoids(k, n_user, dist_arr, balanced, max_iter):
    '''
    Parameters
    ----------
    dist_arr:   array of float [n_user, n_user]

    Returns
    -------
    label:      array of int [n_user]
    inertia:    float
    '''
    label = np.zeros(n_user, dtype=int)
    group_len = int(np.ceil(n_user/k))
    # init
    cen_idx = np.random.choice(n_user, k, replace=False)

    for i in range(max_iter):
        print('iter', i)
        dist = dist_arr[:, cen_idx]
        # kmeans
        if balanced == False:
            new_label = dist.argmin(axis=1)
        # balanced kmeans
        else:
            new_label = np.zeros_like(label)
            label_count = [group_len] * k
            dist_zip = sortArr(dist)
            assigned = []
            for (user_idx, group_idx) in dist_zip:
                if len(assigned) == n_user:
                    break
                if user_idx in assigned:
                    continue
                if label_count[group_idx] > 0:
                    new_label[user_idx] = group_idx
                    assigned.append(user_idx)
                    label_count[group_idx] -= 1
        inertia = np.sum(dist[np.arange(n_user), new_label])

        if (new_label == label).all():
            break
        label = new_label

        # update centroid
        for i in range(k):
            cluster_idx = np.arange(n_user)[new_label == i]
            cluster_dist = dist_arr[cluster_idx, :]
            cluster_inertia = np.sum(cluster_dist, axis=1)
            cen_idx[i] = cluster_idx[cluster_inertia.argmin()]

    return label, inertia

def kmedoids(n_group, n_user, arr, balanced=False, n_init=5, max_iter=10):
    '''
    Parameters
    ----------
    arr:    spdist array [n_user, n_user]
    '''
    tmp_inertia = 1e10
    for i in range(n_init):
        print('init', i, '-------')
        label, inertia = singleKmedoids(n_group, n_user, arr, balanced, max_iter)
        if inertia < tmp_inertia:
            tmp_inertia = inertia
            fin_label = label
    return fin_label

from functools import wraps
import time

def timefn(fn):
    '''compute time cost'''
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        print(f"@time: {t2 - t1: .5f} s")
        return result
    return measure_time

@timefn
def ot_cluster(X, k, max_iters=10):
    # Initialize centroids randomly
    n, _ = X.shape
    centroid = X[np.random.choice(n, size=k, replace=False)]

    # Iterate until convergence or maximum iterations
    for _ in range(max_iters):
        # compute distance
        dist = ((X - centroid[:, np.newaxis])**2).sum(axis=2)  # [k, n]
        inertia = np.min(dist, axis=0).sum()
        
        # compute sinkhorn distance
        lam = 1e-3
        a = np.ones(n) / n
        b = np.ones(k) / k
        trans = ot.emd(a, b, dist.T, lam)

        # Update centroids to the mean of assigned samples
        label = np.argmax(trans, axis=1)
        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])

        # Check if centroids have converged
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid
    print(f'{inertia:.3f}', end=' ')
    return inertia, label#, centroids
