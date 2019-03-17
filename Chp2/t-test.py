import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

model1 = LogisticRegression(solver='liblinear')
result1 = cross_val_score(model1, X, Y, cv=kfold)

model2 = LinearDiscriminantAnalysis()
result2 = cross_val_score(model2, X, Y, cv=kfold)

print("逻辑回归结果：%.3f%% (%.3f%%)" % (result1.mean() * 100, result1.std() * 100))
print("线性判别分析结果：%.3f%% (%.3f%%)" % (result2.mean() * 100, result2.std() * 100))

#t-test
error=result1-result2
T=abs(error.mean()*(10**1/2)/error.std())

alpha=0.05
t_score = stats.t.ppf(1-alpha/ 2, df=num_folds-1)
print(T,t_score)
print('无显著差异' if T<t_score else '有显著差别')

