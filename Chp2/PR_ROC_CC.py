import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score
# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
p=data.groupby('class').size()#[0]/len(data)#数据的分类情况
print(p)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 8
X_train, X_test, Y_traing, Y_test = \
    train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, Y_traing)


def plot_PR(model,x_test,y_test):#绘制PR曲线
    y_pro=model.predict_proba(x_test)
    precision,recall,thresholds=precision_recall_curve(y_test,y_pro[:,1])
    print('阈值为:',thresholds[:10])
    average_precision = average_precision_score(y_test, y_pro[:, 1])
    print('平均准确率为:',average_precision)

    deltaY = precision[:-1] + precision[1:]
    deltaX = recall[:-1] - recall[1:]
    area = deltaX * deltaY / 2

    ax = plt.subplot(111)
    ax.set_title("Precision_Recall Curve AP=%0.5f"%sum(area),verticalalignment='center')

    # plt.step(precision, recall,where='post',alpha=0.2,color='r')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

def plot_ROC(model,x_test,y_test):#绘制ROC和计算AUC，来判断模型的好坏
    y_pro=model.predict_proba(x_test)
    false_positive_rate,recall,thresholds=roc_curve(y_test,y_pro[:,1])
    roc_auc=auc(false_positive_rate,recall)
    ax=plt.subplot(111)
    ax.set_title("Receiver Operating Characteristic Curve",verticalalignment='center')
    plt.plot(false_positive_rate,recall,'b',label='AUC=%0.2f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Recall')
    plt.xlabel('false_positive_rate')
    plt.show()


def Pcost(p,cost01,cost10):
    Pplus=p*cost01/(p*cost01+(1-p)*cost10)
    Pminus=1-Pplus
    return Pplus,Pminus
def plot_CC(model,x_test,y_test):#绘制PR曲线
    y_pro=model.predict_proba(x_test)
    false_positive_rate,recall,thresholds=roc_curve(y_test,y_pro[:,1])
    for i in range(len(false_positive_rate)):
        LP=[0,1]
        RP=[false_positive_rate[i],1-recall[i]]
        plt.plot(LP,RP,'r',linewidth=1)
    plt.plot([0,1],[0,0])
    plt.show()
plot_PR(model,X_test,Y_test)
# plot_ROC(model,X_test,Y_test)
# plot_CC(model,X_test,Y_test)























# predicted=model.predict_proba(X_test)
# precision,recall,thresholds=precision_recall_curve(Y_test,predicted[:,1])
#
# # #PR图
# plt.plot(recall, precision)
# plt.title('Precision/Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.axis('equal')
# plt.show()