import pandas as pd
from scipy import stats
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
n_splits = 1
test_size = 0.33
seed = 8
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

indices1,indices2=[],[]
for train_indices, test_indices in kfold.split(data):
    X_train, X_test, Y_traing, Y_test = \
    X[train_indices],X[test_indices],Y[train_indices],Y[test_indices]

model1 = LogisticRegression(solver='liblinear')
model1.fit(X_train, Y_traing)
predicted1 = model1.predict(X_test)
matrix1 = confusion_matrix(Y_test, predicted1)

model2 = LinearDiscriminantAnalysis()
model2.fit(X_train, Y_traing)
predicted2 = model2.predict(X_test)
matrix2 = confusion_matrix(Y_test, predicted2)

print(classification_report(Y_test,predicted1),matrix1)

print(confusion_matrix(Y_test-predicted1,Y_test-predicted2))

alpha=0.05
chi_score = stats.chi2.ppf(1-alpha,2-1)
print(chi_score)



