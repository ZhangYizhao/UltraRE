import numpy as np

from method.utils import computeRMSE
from method.utils import computeRMSE2


class LinearFiltration(object):
    def __init__(self, param, name='linearFil'):
        self.featureK = param['featureK']

        self.name = name
        self.userMat = []

    def unlearn(self, userMat, var='normal', del_user=[]):
        assert var in ['normal', 'random', 'zero']
        self.userMat = userMat.copy()

        if var == 'normal':
            for id in del_user:
                self.userMat[id-1] = np.zeros(self.featureK)
            for id in del_user:
                self.userMat[id-1] = np.mean(self.userMat, axis=0)
        elif var == 'random':
            for id in del_user:
                self.userMat[id-1] = np.random.randn(self.featureK) / np.sqrt(self.featureK)
        elif var == 'zero':
            for id in del_user:
                self.userMat[id-1] = np.zeros(self.featureK)
        return self.userMat
