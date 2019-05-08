from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def test_Kmeans(*data):
    '''
    测试 KMeans 的用法

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''

    X, labels_true = data
    init = X[[5, 11, 23], :]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(init)
    j = 0
    for i in range(4):
        clst = cluster.KMeans(n_clusters=3, init=init, max_iter=i+1)
        clst.fit(X)
        predicted_labels = clst.predict(X)

        ax[j // 2, j % 2].scatter(X[:, 0], X[:, 1], c=predicted_labels)
        ax[j // 2, j % 2].scatter(clst.cluster_centers_[:, 0], clst.cluster_centers_[:, 1], marker='x', color='red', s=100)
        ax[j // 2, j % 2].set_xlabel('密度')
        ax[j // 2, j % 2].set_ylabel('含糖率')

        j += 1
    plt.show()


data = pd.read_csv('watermelon.txt', header=None,sep=',')
data['y'] = np.zeros((data.shape[0], 1), dtype=int)
data.iloc[8:21, 2] = 1
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values
test_Kmeans(X,y)
