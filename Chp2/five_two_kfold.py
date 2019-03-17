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
num_folds = 2
seed = 8
result1={}
result2={}
result={}
std={}
for i in range(5):
    kfold = KFold(n_splits=num_folds, shuffle=True)

    model1 = LogisticRegression(solver='liblinear')
    r1 = cross_val_score(model1, X, Y, cv=kfold)

    model2 = LinearDiscriminantAnalysis()
    r2 = cross_val_score(model2, X, Y, cv=kfold)

    result1[i]=r1
    result2[i]=r2
    result[i]=r1-r2
    std[i]=sum((r1-r2-(r1+r2)/2)**2)
mu=result[0].mean()
sigma=sum(std.values())
print(result1)
print(result2)
#t-test
T=mu/((0.2*sigma)**0.5)
alpha=0.05
t_score = stats.t.ppf(1-alpha/ 2, df=5)
print(T,t_score)
print('无显著差异' if abs(T)<t_score else '有显著差别')

