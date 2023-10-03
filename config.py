from os.path import exists
from os import mkdir
import numpy as np
import time
import warnings
import os

from read import readRating, loadData, readSparseMat, get_gender
from read import RatingData
from method.utils import saveObject, loadObject
from method.scratch import Scratch
from method.sequential import Sequential
from method.sisa import Sisa
from group import Group, DATA_DIR, SAVE_DIR


class InsParam(object):
    def __init__(self, dataset='toy', epochs=50, n_worker=24, layers=[32], n_group=2, del_per=2, del_type='test', dis_type='nor'):
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
        self.dis_type = dis_type  # type of distinguishiability loss

        # dataset-varied param
        self.del_rating = []  # 2d array/list [[uid, iid], ...]
        self.dataset = dataset
        self.max_rating = 5
        self.del_per = del_per
        self.del_type = del_type

        if dataset == 'toy':
            self.train_dir = DATA_DIR + '/toy/0_train.csv'
            self.test_dir = DATA_DIR + '/toy/0_test.csv'
            self.n_user = 1509
            self.n_item = 2072
            self.del_user = [3]
        elif dataset == 'db':
            self.train_dir = DATA_DIR + '/db/squ0_train.csv'
            self.test_dir = DATA_DIR + '/db/squ0_test.csv'
            self.n_user = 108662
            self.n_item = 28
            if self.del_type == 'test':
                self.del_user = [11334, 5469]#, 5471, 3122, 3114, 13473, 9075, 28121, 9052, 9051]
            else:
                np.random.seed(0)
                self.del_user = np.random.choice(self.n_user, 5000, replace=False)
        elif dataset == 'ad':
            self.train_dir = DATA_DIR + '/ad/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ad/squ0_test.csv'
            self.n_user = 5541
            self.n_item = 3568
            if self.del_type == 'test':
                self.del_user = [240, 88]#, 79, 334, 220, 186, 27, 332, 532, 1022]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [240,   88,   79,  334,  220,  186,   27,  332,  532, 1022,  140, 381,  131,  155,  935,  628, 2738,   17,   90,  145,  470,  608, 373,  128, 1316, 1478, 1456,  134,  236, 3185]
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = idx[:n_del]
            self.epochs = 20
        elif dataset == 'adn':
            self.train_dir = DATA_DIR + '/adn/squ0_train.csv'
            self.test_dir = DATA_DIR + '/adn/squ0_test.csv'
            self.n_user = 5541
            self.n_item = 3568
            if self.del_type == 'test':
                self.del_user = [240, 88]#, 79, 334, 220, 186, 27, 332, 532, 1022]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [240,   88,   79,  334,  220,  186,   27,  332,  532, 1022,  140, 381,  131,  155,  935,  628, 2738,   17,   90,  145,  470,  608, 373,  128, 1316, 1478, 1456,  134,  236, 3185]
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = idx[:n_del]
            self.epochs = 20
        elif dataset == 'am':
            self.train_dir = DATA_DIR + '/am/squ0_train.csv'
            self.test_dir = DATA_DIR + '/am/squ0_test.csv'
            self.n_user = 123960
            self.n_item = 50052
            if self.del_type == 'test':
                self.del_user = [1111, 844]#, 5153, 1397, 828, 1260, 341, 2753, 6237, 2447]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 300 if self.del_per == 2 else 600
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [ 1111,   844,  5153,  1397,   828,  1260,   341,  2753,  6237, 2447,  1272,   716,  2176,  2175,  5860,  2536,   688,  2370, 1392,  2856,  1181,   772,  4984,  4289,  2323,    21,  2481, 598,   864,  3082,   524,  2187,  2699,  2267,  2155,  2179, 2489,  1231,   874,  1122, 57758,  1400,  2314,  1011,   582, 2363,  2067,   399,   435,  7752,   321,  1993,   954,  8306, 5199,   907,   625,  2454,   717,  3421,  2468,  3015,  3017, 1049,  2659,  2205,  1516,  1276,  1523,  4488,  1003,  4790, 914,  3881,  2057,  2385, 13201,  7655,  4870,  1102,  8536, 1078,  1019,  4130,   270,  5820,  2377,  4652,  7117,  5485, 1337,  2836,  7015,  2569,  4167,  4947,  1246,  1258,  3935, 6885,  6573,  2544,  2965,  4890,  2689,  4373,  1316,   112, 1898,  1134,  6029,   680,  2199,  3444,  2811, 12675,  3033, 2721,  2461,  4792,  2374,  6790,  5615,  2833, 14356,  4323, 3835,  2129,  4789,  9989,  6614,  1833,   747,  1363,  7344, 8544,  1644,  4495,  8278,  4472,  1926,   457,  5122,   224, 4854,  6704,   528, 11305,  6999, 11580,  4809,  2559,  5276, 1006,  4659,  7736,   838,  2442,  5217,  8181, 15427,  4550, 257,  2352,  4664,  1490,  1325,  8869,  4524,  3445,   125, 368,   805,  1022,  7346,  8966,  2371,  6495,  6625,   391, 511,  1066, 18185,  4497,  1067,  3726,  2251,  6880,  9399, 469,  1109,  1472,  2809,  3016,  1244, 11082, 14241,  3509, 1187,  5417,  5712, 15745,  2723,   730,  9994,  9316,   583, 2803,  5946,  6789, 22355,  4322,  3025,  9682,  3560, 16770, 2470,  4503,  2642,  6838,  2616,  2892, 370, 27530,   379, 1662,   302,  2395,  5911,  2390,  9022,  9264,  5515, 18075, 840,  5905,   689,  1345,  6670,  2513,   977,  8639, 12454, 988,   337,  1444,  4840,   340,  4442,  3796,  7560,   247, 2746,   427,  1432,  1110,   902, 16061,  2713,  4262, 16050, 14334,  4432,  2561,  1105, 11984,  5749,   595,  6098,  3852, 1054, 12444,  9855,  2675,  6518,  2914,   933, 25600,   413, 4253,  4478, 13630,   944,  3034,  2214,  1117,  4915,  2399, 5279,  8286,  1690,  1080,  1618,   434, 27193,  5974,  3088, 2320, 16072,  8983]
                n_del = 150 if self.del_per == 2 else 300
                self.del_user = idx[:n_del]
                # self.del_user = [ 1111,   844,  5153,  1397,   828,  1260,   341,  2753,  6237,\n        2447,  1272,   716,  2176,  2175,  5860,  2536,   688,  2370,\n        1392,  2856,  1181,   772,  4984,  4289,  2323,    21,  2481,\n         598,   864,  3082,   524,  2187,  2699,  2267,  2155,  2179,\n        2489,  1231,   874,  1122, 57758,  1400,  2314,  1011,   582,\n        2363,  2067,   399,   435,  7752,   321,  1993,   954,  8306,\n        5199,   907,   625,  2454,   717,  3421,  2468,  3015,  3017,\n        1049,  2659,  2205,  1516,  1276,  1523,  4488,  1003,  4790,\n         914,  3881,  2057,  2385, 13201,  7655,  4870,  1102,  8536,\n        1078,  1019,  4130,   270,  5820,  2377,  4652,  7117,  5485,\n        1337,  2836,  7015,  2569,  4167,  4947,  1246,  1258,  3935,\n        6885,  6573,  2544,  2965,  4890,  2689,  4373,  1316,   112,\n        1898,  1134,  6029,   680,  2199,  3444,  2811, 12675,  3033,\n        2721,  2461,  4792,  2374,  6790,  5615,  2833, 14356,  4323,\n        3835,  2129,  4789,  9989,  6614,  1833,   747,  1363,  7344,\n        8544,  1644,  4495,  8278,  4472,  1926,   457,  5122,   224,\n        4854,  6704,   528, 11305,  6999, 11580,  4809,  2559,  5276,\n        1006,  4659,  7736,   838,  2442,  5217,  8181, 15427,  4550,\n         257,  2352,  4664,  1490,  1325,  8869,  4524,  3445,   125,\n         368,   805,  1022,  7346,  8966,  2371,  6495,  6625,   391,\n         511,  1066, 18185,  4497,  1067,  3726,  2251,  6880,  9399,\n         469,  1109,  1472,  2809,  3016,  1244, 11082, 14241,  3509,\n        1187,  5417,  5712, 15745,  2723,   730,  9994,  9316,   583,\n        2803,  5946,  6789, 22355,  4322,  3025,  9682,  3560, 16770,\n        2470,  4503,  2642,  6838,  2616,  2892,   370, 27530,   379,\n        1662,   302,  2395,  5911,  2390,  9022,  9264,  5515, 18075,\n         840,  5905,   689,  1345,  6670,  2513,   977,  8639, 12454,\n         988,   337,  1444,  4840,   340,  4442,  3796,  7560,   247,\n        2746,   427,  1432,  1110,   902, 16061,  2713,  4262, 16050,\n       14334,  4432,  2561,  1105, 11984,  5749,   595,  6098,  3852,\n        1054, 12444,  9855,  2675,  6518,  2914,   933, 25600,   413,\n        4253,  4478, 13630,   944,  3034,  2214,  1117,  4915,  2399,\n        5279,  8286,  1690,  1080,  1618,   434, 27193,  5974,  3088,\n        2320, 16072,  8983,   931,   350,   652,  6738,  4651,  1307,\n        4909,   739,   397,  6190,   472,   167, 38975,  5599,  8399,\n        4902,  6451,  2073,  3163,  3464,  6080,  5270,  1365,  8672,\n        4390,  8326,   280,   241,  2883,  5553,  3683,   798,  5490,\n        9951,  2243,  1383,  1169, 26488,  4908,  7780,  5730,  6420,\n        8035,  9023,  6597,  3072, 93206,  7897,   404,  3693,  6688,\n        2968,  1099,  4835,  2528,  1136,  9306,  2760,   928,  9577,\n        5775,  2218, 25126,  8630,  2657,  1215,   596,  5606, 10417,\n        5227,  5809, 16166,  9958,   619,  5563,  1344,  4724,  2355,\n        1176,  4823,   121,  2531,  5788,  5414,  3023, 10044,  7514,\n        2708,  2041, 11006, 16699,  5521, 37059,  2814,  4429,   843,\n       38299,  3575,  4719,  4476, 27284,  5189,  5280,  4737,  3564,\n         515,  5249,  6587,   715,  4482,  8315,  2651, 22414, 15979,\n        4437,  2357,  1050,  5012,   352,  8987, 10414, 13222,  4710,\n       10584,  1988,  6940, 16419,  1280,  6810,  8738,  5086,  1843,\n         380,  3648,  6846, 14013,  5244,  7055,   956,  3563,  2923,\n        4583, 14342,  6769,  2864,  5907,   291,  5372,  8412,  5295,\n        5347,  6994,  2165,  6165,  6242,  6631, 22457,  8766, 10547,\n        1330,  5595,  7009,  3204,  4976, 19144, 16098,  1599,  2759,\n       10522, 19194,  2679,   450,   520, 11037,  2932,  6411, 10805,\n       16041,   560,  1341,   400, 11311,  4920,  9194,  6926,  4939,\n       10233, 10777,  9380,   801,  4813,  9492,  9538,  5009, 10992,\n        6280, 13041,  6436,  5870,  1343,     7, 19357,  6525,   248,\n        4812,  8924,  7322,  2851,  7210,  1373, 12148,  2660,  6651,\n        6206,  3451,  2863,   266,  4637,   225, 55506,   778, 14193,\n        2716,   387,  4903, 37028,  8524,   919,  2430,  1069,  8994,\n        1236, 20599, 10154,   462, 13272,  5052, 10239,  8267, 10107,\n       13986,  4845,   616,  8868,  5016,  8202,  9937,  3561,  9084,\n        3621,  9636,   807,  3654, 59374,     6,   542,  2815, 22370,\n        5201,   578,  1209,  5518,  5108,  5829,  1031,  1352, 10348,\n       22654,  1347,  7260, 13309,  6818, 10176,  2880,   328,  6056,\n       18104,  3795,  3760,  3682, 14969,  2747,  4573,  8748,  8733,\n       11693,  7690, 22374,  6557,  1329,  5299,  6558,  6189,  2664,\n        7705,  7600,   153, 63017,  9592,  4003]
            self.epochs = 20
            self.lr = 0.0001
        elif dataset == 'ah':
            self.train_dir = DATA_DIR + '/ah/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ah/squ0_test.csv'
            self.n_user = 38609
            self.n_item = 18534
            if self.del_type == 'test':
                self.del_user = [873, 10721]#, 172, 2429, 5222, 5924, 10676, 1201, 108, 10699]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [  873, 10721,   172,  2429,  5222,  5924, 10676,  1201,   108, 10699, 10597,    31, 10923,   276, 18737, 10715,  4918,   588, 1386,  7244,  5541, 12409,  3811,  4971, 10671,   192,  8715, 643,   396,  1220]
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = idx[:n_del]
            self.epochs = 20
        elif dataset == 'ml1':
            self.train_dir = DATA_DIR + '/ml1/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ml1/squ0_test.csv'
            self.n_user = 6040
            self.n_item = 3416
            if self.del_type == 'test':
                self.del_user = [4168, 1679]#, 4276, 1940, 1180, 888, 3617, 2062, 1149, 1014]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [4168, 1679, 4276, 1940, 1180,  888, 3617, 2062, 1149, 1014, 5794, 4343, 1979, 2908, 1448, 4509,  423, 4226, 5830, 3390, 3840, 4507, 1087, 5366, 3807,  548, 1284, 3223, 3538, 4542]
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = idx[:n_del]
            self.epochs = 50
        elif dataset == 'ml1m':
            self.train_dir = DATA_DIR + '/ml1m/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ml1m/squ0_test.csv'
            self.n_user = 6040
            self.n_item = 3416
            if self.del_type == 'test':
                self.del_user = [4168, 1679]#, 4276, 1940, 1180, 888, 3617, 2062, 1149, 1014]
            elif self.del_type == 'rand':
                np.random.seed(0)
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = np.random.choice(self.n_user, n_del, replace=False)
            elif self.del_type == 'top':
                idx = [4168, 1679, 4276, 1940, 1180,  888, 3617, 2062, 1149, 1014, 5794, 4343, 1979, 2908, 1448, 4509,  423, 4226, 5830, 3390, 3840, 4507, 1087, 5366, 3807,  548, 1284, 3223, 3538, 4542]
                n_del = 15 if self.del_per == 2 else 30
                self.del_user = idx[:n_del]
            self.epochs = 50
            # self.attr = get_gender(DATA_DIR + '/ml1n/user_gender.npy')
        elif dataset == 'ml20':
            self.train_dir = DATA_DIR + '/ml20/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ml20/squ0_test.csv'
            self.n_user = 138493
            self.n_item = 26744
            self.del_user = [118204, 8404]#, 82417, 121534, 125793, 74141, 34575, 131903, 83089, 59476]
        elif dataset == 'ml25':
            self.train_dir = DATA_DIR + '/ml25/squ0_train.csv'
            self.test_dir = DATA_DIR + '/ml25/squ0_test.csv'
            self.n_user = 162541
            self.n_item = 59047
            self.del_user = [72314, 80973]#, 137292, 33843, 20054, 109730, 92045, 49402, 30878, 115101]
        elif dataset == 'netf':
            self.train_dir = DATA_DIR + '/netf/squ0_train.csv'
            self.test_dir = DATA_DIR + '/netf/squ0_test.csv'
            self.n_user = 480189
            self.n_item = 17770
            self.del_user = [131, 169]#, 42, 84, 166, 179, 193, 302, 2809, 687]
    
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
        if learn_type in ['seq', 'sisa']:
            for i in range(n_group):
                if i == 0:
                    test_total = test_rating[i].copy()
                else:
                    test_total = np.hstack((test_total, test_rating[i]))
                train_dlist.append(loadData(RatingData(train_rating[i]), self.param.batch, self.param.n_worker))
                test_dlist.append(loadData(RatingData(test_rating[i]), self.param.batch, self.param.n_worker, False))
            test_data = loadData(RatingData(test_total), self.param.batch, self.param.n_worker, False)
        elif learn_type == 'add':
            for i in range(n_group):
                if i == 0:
                    train_cur = train_rating[i].copy()
                    test_cur = test_rating[i].copy()
                else:
                    train_cur = np.hstack((train_cur, train_rating[i]))
                    test_cur = np.hstack((test_cur, test_rating[i]))
                train_dlist.append(loadData(RatingData(train_cur), self.param.batch, self.param.n_worker))
                test_dlist.append(loadData(RatingData(test_cur), self.param.batch, self.param.n_worker, False))
            test_data = loadData(RatingData(test_cur), self.param.batch, self.param.n_worker, False)

        # mkdir
        if is_save ==True:
            save_dir = SAVE_DIR + self.name + '/' + saving_name
            if exists(save_dir) == False:
                mkdir(save_dir)
        else:
            save_dir = ''

        # train model
        if learn_type in ['seq', 'add']:
            model = Sequential(self.param, model_type, n_group, train_index)
        elif learn_type == 'sisa':
            model = Sisa(self.param, model_type, n_group, train_index)
        if is_del == False:
            model.learn(train_dlist, test_dlist, test_data, verbose, save_dir)
        else:
            del_user = self.param.del_user
            for rating in self.param.del_rating:
                if rating[0] not in del_user:
                    del_user.append(rating[0])
            model.unlearn(model_list, train_dlist, test_dlist, test_data, del_user, verbose, save_dir)
            
        print('End of training', self.name, saving_name)
        print()

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
        # self.__full(is_save, 'MF_retrain', 'mf', True, verbose)

        '''model DMF'''
        # full train without deletion
        # self.__full(is_save, 'DMF_full_train', 'dmf', False, verbose)

        # retrain from scratch after deletion
        # self.__full(is_save, 'DMF_retrain', 'dmf', True, verbose)

        '''model NMF'''
        # full train without deletion
        # self.__full(is_save, 'NMF_full_train', 'nmf', False, verbose)

        # retrain from scratch after deletion
        # self.__full(is_save, 'NMF_retrain', 'nmf', True, verbose)

        '''test model'''
        # self.__full(is_save, 'GMF_full_train', 'gmf', False, verbose)

    def runGroup(self, is_save=True, learn_type='seq', group_type='uniform', n_group=5, verbose=1):
        # '''model MF'''
        # # seq learn with deletion
        # saving_name = 'MF_' + group_type + '_' + learn_type + '_learn'
        # model_list = self.__group([], is_save, learn_type, saving_name, 'mf', 
        #                             False, group_type, n_group, verbose)

        # # seq unlearn with deletion
        # saving_name = 'MF_' + group_type + '_' + learn_type + '_unlearn'
        # self.__group(model_list, is_save, learn_type, saving_name, 'mf', 
        #                 True, group_type, n_group, verbose)

        '''model DMF'''
        # # seq learn with deletion
        # saving_name = 'DMF_' + group_type + '_' + learn_type + '_learn'
        # model_list = self.__group([], is_save, learn_type, saving_name, 'dmf', 
        #                             False, group_type, n_group, verbose)

        # # seq unlearn with deletion
        # saving_name = 'DMF_' + group_type + '_' + learn_type + '_unlearn'
        # self.__group(model_list, is_save, learn_type, saving_name, 'dmf', 
        #                 True, group_type, n_group, verbose)

        '''model NMF'''
        # seq learn with deletion
        saving_name = 'NMF_' + group_type + '_' + learn_type + '_learn'
        model_list = self.__group([], is_save, learn_type, saving_name, 'nmf', 
                                    False, group_type, n_group, verbose)

        # seq unlearn with deletion
        saving_name = 'NMF_' + group_type + '_' + learn_type + '_unlearn'
        self.__group(model_list, is_save, learn_type, saving_name, 'nmf', 
                        True, group_type, n_group, verbose)

        '''test model'''
        # saving_name = 'GMF_' + group_type + '_' + learn_type + '_learn'
        # model_list = self.__group([], is_save, learn_type, saving_name, 'gmf', 
        #                             False, group_type, n_group, verbose)
