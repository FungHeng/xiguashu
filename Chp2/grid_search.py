# -*- coding: utf-8 -*-
"""
    模型选择
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    参数优化

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn.datasets import load_digits
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import scipy

def test_GridSearchCV():
    '''
    测试 GridSearchCV 的用法。使用 LogisticRegression 作为分类器，主要优化 C、penalty、multi_class 等参数

    :return: None
    '''
    ### 加载数据
    digits = load_digits()
    X_train,X_test,y_train,y_test=train_test_split(digits.data, digits.target,test_size=0.25,
                random_state=0,stratify=digits.target)
    #### 参数优化 ######
    tuned_parameters = [{'penalty': ['l1','l2'],
                        'C': [0.01,0.05,0.1,0.5,1,5,10,50,100],
                        'solver':['liblinear'],
                        'multi_class': ['ovr']},

                        # {'penalty': ['l2'],
                        # 'C': [0.01,0.05,0.1,0.5,1,5,10,50,100],
                        #  'solver':['lbfgs'],
                        # 'multi_class': ['ovr','multinomial']},
                        ]
    clf=GridSearchCV(LogisticRegression(max_iter=2000,tol=1e-6),tuned_parameters,cv=2)
    clf.fit(X_train,y_train)
    print("Best parameters set found:",clf.best_params_)
    print("Grid scores:")

    # params, mean_train_score, std_train_score=clf.cv_results_['params'],clf.cv_results_['mean_train_score'],clf.cv_results_['std_train_score']
    #
    # for mean, std, params in zip(mean_train_score, std_train_score, params):
    #     print("\t%.3f (std:%.3f) for %s" % (mean, std , params))
    #
    # print("Optimized Score:",clf.score(X_test,y_test))
    # print("Detailed classification report:")
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
def test_RandomizedSearchCV():
    '''
    测试 RandomizedSearchCV 的用法。使用 LogisticRegression 作为分类器，主要优化 C、multi_class 等参数。其中 C 的分布函数为指数分布

    :return:  None
    '''
    ### 加载数据
    digits = load_digits()
    X_train,X_test,y_train,y_test=train_test_split(digits.data, digits.target,
                test_size=0.25,random_state=0,stratify=digits.target)
    #### 参数优化 ######
    tuned_parameters ={  'C': scipy.stats.expon(scale=100), # 指数分布
                        'multi_class': ['ovr','multinomial']}

    clf=RandomizedSearchCV(LogisticRegression(penalty='l2',solver='newton-cg',tol=1e-6,max_iter=100),tuned_parameters,cv=2,scoring="accuracy",n_iter=10)
    clf.fit(X_train,y_train)
    print("Best parameters set found:",clf.best_params_)
    print("Randomized Grid scores:")

    params, mean_train_score, std_train_score = clf.cv_results_['params'], clf.cv_results_['mean_train_score'], clf.cv_results_['std_train_score']
    for i in range(len(params)):
        print("\t%0.3f (std:%0.03f) for %s" % (mean_train_score[i], std_train_score[i] , params[i]))

    print("Optimized Score:",clf.score(X_test,y_test))
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

if __name__=='__main__':
    test_GridSearchCV()# 调用 test_GridSearchCV
    # test_RandomizedSearchCV() # 调用 test_RandomizedSearchCV
