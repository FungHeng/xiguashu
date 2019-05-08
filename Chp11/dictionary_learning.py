# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    字典学习

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn.decomposition import DictionaryLearning
import numpy as np


def test_DictionaryLearning():
    '''
    测试 DictionaryLearning 的用法

    :return: None
    '''
    X = np.array([[1, 1, 1, 1, 0, 0],
                  [2, 2, 0, 0, 0, 0],
                  [3, 3, 0, 0, 0, 0],
                  [0, 0, 3, 3, 1, 1],
                  [0, 0, 0, 0, 2, 2]])


    print("before transform:\n%s" % X)
    dct=DictionaryLearning(n_components=3)
    dct.fit(X.T)
    D = dct.components_.T
    print("components is:\n", D)
    print("after transform:\n", dct.transform(X.T).T)
    print("restruction:\n", D.dot(dct.transform(X.T).T))

    test = np.array([[0, 1, 1],
                     [0, 2, 1],
                     [0, 3, 1],
                     [1, 0, 1],
                     [1, 0, 1]])
    print("test after transform:\n", dct.transform(test.T).T)
    print("restruction:\n", D.dot(dct.transform(test.T).T))

if __name__=='__main__':
    test_DictionaryLearning()  # 调用 test_DictionaryLearning
