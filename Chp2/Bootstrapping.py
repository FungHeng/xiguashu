import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
# print(data.shape)
# 将数据分为输入数据和输出结果
seed=8
train = data.sample(n=data.shape[0],replace=True)#,random_state=seed   #replace表示允许抽样重复
# print(train.head())
# print(train.index.sort_values())

X_train,Y_traing =train.iloc[:,:8],train.iloc[:,8]
print(X_train)
test = data.loc[data.index.difference(train.index)]
X_test, Y_test=test.iloc[:,:8],test.iloc[:,8]

print(X_test.shape[0]/data.shape[0])
# model = LogisticRegression(solver='newton-cg')
# # model = LogisticRegression(solver='liblinear')
# model.fit(X_train, Y_traing)
# result = model.score(X_test, Y_test)
# print("算法评估结果：%.3f%%" % (result * 100))
