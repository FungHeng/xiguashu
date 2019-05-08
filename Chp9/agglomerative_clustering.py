# -*- coding: utf-8 -*-
"""
    聚类和EM算法
    ~~~~~~~~~~~~~~~~

    AgglomerativeClustering

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def test_AgglomerativeClustering(*data,n=2):
    '''
    测试 AgglomerativeClustering 的用法

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    clst=cluster.AgglomerativeClustering(n_clusters=n,linkage='complete')
    predicted_labels=clst.fit_predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))

    labels = clst.labels_
    n_clusters_ = len(set(labels))  # 获取分簇的数目
    for i in range(n_clusters_):
        one_cluster = X[labels == i]
        plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o', label=i)
    plt.legend()
    plt.show()

def test_AgglomerativeClustering_nclusters(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随 n_clusters 参数的影响

    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=[2]
    ARIs=[]
    for num in nums:
        clst=cluster.AgglomerativeClustering(n_clusters=num)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))

    print(ARIs)
    ## 绘图
    # fig=plt.figure()
    # ax=fig.add_subplot(1,1,1)
    # ax.plot(nums,ARIs,marker="+")
    # ax.set_xlabel("n_clusters")
    # ax.set_ylabel("ARI")
    # fig.suptitle("AgglomerativeClustering")
    # plt.show()
def test_AgglomerativeClustering_linkage(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随链接方式的影响

    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    linkages=['ward','complete','average']
    markers="+o*"
    for i, linkage in enumerate(linkages):
        ARIs=[]
        for num in nums:
            clst=cluster.AgglomerativeClustering(n_clusters=num,linkage=linkage)
            predicted_labels=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        ax.plot(nums,ARIs,marker=markers[i],label="linkage:%s"%linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

def dendrogram(data):
    disMat = sch.distance.pdist(data,'euclidean')
    #进行层次聚类:
    Z=sch.linkage(disMat,method='complete')
    #将层级聚类结果以树状图表示出来
    sch.dendrogram(Z)
    plt.axhline(0.22,ls='--',c='k')
    plt.show()

if __name__== '__main__':
    data = pd.read_csv('watermelon.txt', header=None, sep=',')
    data['y'] = np.zeros((data.shape[0], 1), dtype=int)
    data.iloc[8:21, 2] = 1
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    for i in range(2, 3):
        test_AgglomerativeClustering(X, y,n=i)
    # test_AgglomerativeClustering_nclusters(X, y)
    # dendrogram(X)
    test_AgglomerativeClustering_nclusters(X, y)