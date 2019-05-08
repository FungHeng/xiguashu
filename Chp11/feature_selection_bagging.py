# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    包裹式特征选择

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_iris
from  sklearn import  model_selection
from sklearn.preprocessing import LabelEncoder

def test_RFE(*data):
    '''
    测试 RFE 的用法，其中目标特征数量为 2

    :return: None
    '''
    X ,y = data
    estimator = LinearSVC(max_iter=4000)
    selector = RFE(estimator=estimator, n_features_to_select=3)
    selector.fit(X, y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
def test_RFECV(*data):
    '''
    测试 RFECV 的用法

    :return:  None
    '''
    X ,y = data
    estimator=LinearSVC(max_iter=4000)
    selector=RFECV(estimator=estimator,cv=5)
    selector.fit(X,y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
    print("Grid Scores %s"%selector.grid_scores_)
def test_compare_with_no_feature_selection(*data):
    '''
    比较经过特征选择和未经特征选择的数据集，对 LinearSVC 的预测性能的区别

    :return: None
    '''
    ### 加载数据
    X ,y = data
    ### 特征提取
    estimator=LinearSVC(max_iter=4000)
    selector=RFE(estimator=estimator,n_features_to_select=2)
    X_t=selector.fit_transform(X,y)
    #### 切分测试集与验证集
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X, y,
                test_size=0.25,random_state=0,stratify=y)
    X_train_t,X_test_t,y_train_t,y_test_t=model_selection.train_test_split(X_t, y,
                test_size=0.25,random_state=0,stratify=y)
    ### 测试与验证
    clf=LinearSVC(max_iter=4000)
    clf_t=LinearSVC(max_iter=4000)
    clf.fit(X_train,y_train)
    clf_t.fit(X_train_t,y_train_t)
    print("Original DataSet: test score=%s"%(clf.score(X_test,y_test)))
    print("Selected DataSet: test score=%s"%(clf_t.score(X_test_t,y_test_t)))
if __name__=='__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    # test_RFE(X,y) # 调用 test_RFE
    test_RFECV(X,y) # 调用 test_RFECV
    # test_compare_with_no_feature_selection(X,y) # 调用 test_compare_with_no_feature_selection

    # import pandas as pd
    # data = pd.read_csv('watermelon3_0_Ch.csv', encoding='gbk')[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']]
    # le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
    # for col in data.columns[:6]:  # 为每一列序列化
    #     data[col] = le.fit_transform(data[col])
    #
    # X = data.iloc[:, :8].values.astype(float)
    # y = le.fit_transform(data.iloc[:, 8]).astype(int)
    # test_RFECV(X, y)
    # print(X)