# -*- coding: utf-8 -*-
"""
    kNN和降维
    ~~~~~~~~~~

    MDS

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,manifold,metrics,utils

def load_data():
    '''
    加载用于降维的数据

    :return: 一个元组，依次为训练样本集和样本集的标记
    '''
    iris=datasets.load_iris()  # 使用 scikit-learn 自带的 iris 数据集
    return  iris.data,iris.target

def test_MDS(*data):
    '''
    测试 MDS 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、训练样本的标记
    :return: None
    '''
    X,y=data
    for n in [4,3,2,1]: # 依次考察降维目标为 4维、3维、2维、1维
        mds=manifold.MDS(n_components=n)
        mds.fit(X)
        print('stress(n_components=%d) : %.2f'% (n,mds.stress_))  # 不一致的距离总和

def plot_MDS(*data):
    '''
    绘制经过 使用 MDS 降维到二维之后的样本点

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、训练样本的标记
    :return: None
    '''
    X,y=data
    mds=manifold.MDS(n_components=2, metric=True)
    X_r=mds.fit_transform(X) #原始数据集转换到二维
    # print(mds.dissimilarity_matrix_.shape)
    # print(mds.embedding_)
    # W = X_r.T.dot(X).dot(np.linalg.pinv(X.T.dot(X)))
    # print(W.dot(X.T).T)

    ### 绘制二维图形
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)# 颜色集合，不同标记的样本染不同的颜色
    for label ,color in zip( np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label,color=color)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()

def MDS1(X,d):
    D = metrics.euclidean_distances(X)
    DSquare = D ** 2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis=0)
    rowMean = np.mean(DSquare, axis=1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)  # 求特征值及特征向量
    X_r = np.dot(eigVec[:,:2].real, np.sqrt(np.diag(eigVal[:2]).real))

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)# 颜色集合，不同标记的样本染不同的颜色
    for label, color in zip(np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label,color=color)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()



if __name__=='__main__':
    X,y=load_data() # 产生用于降维的数据集
    # test_MDS(X,y)   # 调用 test_MDS
    # plot_MDS(X,y)   # 调用 plot_MDS
    MDS1(X, 2)

