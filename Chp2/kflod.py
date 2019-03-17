import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)

# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
seed = 8
kfold = KFold(n_splits=num_folds, random_state=seed)

# kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)#分层采样
model = LogisticRegression(solver='liblinear')
result = cross_val_score(model, X, Y, cv=kfold)
print(result)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))
