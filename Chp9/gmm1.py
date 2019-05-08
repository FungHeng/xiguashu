import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture


def gmm(*data):
    X, labels_true=data
    mean_init = X[[5,21,26],:]
    precision_init = np.array([[[10,0],[0,10]],[[10,0],[0,10]],[[10,0],[0,10]]])
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    plt.rcParams['font.sans-serif']=['SimHei']
    j=0
    for i in [5,10,20,50]:
        clst = mixture.GaussianMixture(n_components=3, weights_init=[1 / 3, 1 / 3, 1 / 3], means_init=mean_init,warm_start=True, max_iter=i, precisions_init=precision_init)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ax[j // 2,j % 2].scatter(X[:,0], X[:,1], c=predicted_labels)
        ax[j // 2, j % 2].scatter(clst.means_[:, 0], clst.means_[:, 1], marker='x', color='red', s=100)
        ax[j // 2, j % 2].set_xlabel('密度')
        ax[j // 2, j % 2].set_ylabel('含糖率')

        j += 1
    plt.show()


data = pd.read_csv('watermelon.txt', header=None,sep=',')
data['y'] = np.zeros((data.shape[0], 1), dtype=int)
data.iloc[8:21, 2] = 1
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values
gmm(X,y)


