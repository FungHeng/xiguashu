# -*- coding: utf-8 -*-
"""
    支持向量机
    ~~~~~~~~~~~~~~~~

    LinearSVC

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,model_selection,svm

def load_data_classfication():
    '''
    加载用于分类问题的数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    iris=datasets.load_iris() # 使用 scikit-learn 自带的 iris 数据集
    X_train=iris.data[:,:3]
    y_train=iris.target[:]
    # X_train=iris.data[50:,:3]
    # y_train=iris.target[50:]
    return model_selection.train_test_split(X_train, y_train,test_size=0.25,
		random_state=0,stratify=y_train) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

def test_LinearSVC(*data):
    '''
    测试 LinearSVC 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    cls=svm.LinearSVC(max_iter=4000)
    cls.fit(X_train,y_train)
    print('Coefficients:\n%s\nintercept:\n%s'%(cls.coef_,cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))
    plot_data(X_test,y_test,coef=cls.coef_,intercept=cls.intercept_)

def plot_data(*data,coef,intercept):
    X,y=data
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y)

    Xmin,Xmax=np.min(X[:,0]),np.max(X[:,0])
    Ymin,Ymax=np.min(X[:,1]),np.max(X[:,1])
    Xaxis=np.linspace(Xmin,Xmax,50)
    Yaxis=np.linspace(Ymin,Ymax,50)
    Xaxis,Yaxis=np.meshgrid(Xaxis,Yaxis)
    for i in range(3):
        Zaxis=(-intercept[i]-coef[i,0]*Xaxis-coef[i,1]*Yaxis)/coef[i,2]
        ax.plot_surface(Xaxis, Yaxis, Zaxis)

    # Zaxis = (-intercept - coef[0, 0] * Xaxis - coef[0, 1] * Yaxis) / coef[0, 2]
    # ax.plot_surface(Xaxis, Yaxis, Zaxis)
    plt.show()

def test_LinearSVC_loss(*data):
    '''
    测试 LinearSVC 的预测性能随损失函数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    losses=['hinge','squared_hinge']
    for loss in losses:
        cls=svm.LinearSVC(loss=loss,max_iter=8000)
        cls.fit(X_train,y_train)
        print("Loss:%s"%loss)
        print('Coefficients:\n%s,\n intercept:\n%s' % (cls.coef_, cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))
def test_LinearSVC_L12(*data):
    '''
    测试 LinearSVC 的预测性能随正则化形式的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    L12=['l1','l2']
    for p in L12:
        cls=svm.LinearSVC(penalty=p,dual=False,max_iter=3000)
        cls.fit(X_train,y_train)
        print("penalty:%s"%p)
        print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))
def test_LinearSVC_C(*data):
    '''
    测试 LinearSVC 的预测性能随参数C的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:   None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,1)
    train_scores=[]
    test_scores=[]
    for C in Cs:
        cls=svm.LinearSVC(C=C,max_iter=40000)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,train_scores,label="Traing score")
    ax.plot(Cs,test_scores,label="Testing score")
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LinearSVC")
    ax.legend(loc='best')
    plt.show()
if __name__=="__main__":
    X_train,X_test,y_train,y_test=load_data_classfication() # 生成用于分类的数据集
    test_LinearSVC(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC
    # test_LinearSVC_loss(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_loss
    # test_LinearSVC_L12(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_L12
    # test_LinearSVC_C(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_C