import numpy as np
from scipy import stats
import matplotlib.pylab as plt
def calcTX2(N, k,r):
    partone=(12*N)/(k*(k+1))
    parttwo=sum(r**2)-k*(k+1)*(k+1)/4
    return partone*parttwo

def TF(TX2,N,k):
    return ((N-1)*TX2)/(N*(k-1)-TX2)

def calcCD(qAlpha,N,k):
    return qAlpha*np.sqrt(k*(k+1)/(6*N))

def Nemenyi(TF,F_value,k):
    if TF<=F_value:
        print('所有算法性能相同')
    else:
        print('算法的性能显著不同')
        CD=calcCD(2.344,N,k)
        l1=r-CD/2
        r1=r+CD/2
        plt.figure()
        ax=plt.gca()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        ax.set_xticks([0,1,2,3])
        ax.set_yticks(range(1,k+1))
        ax.set_yticklabels(['算法1','算法2','算法3'])
        colors='brg'
        for i,color in zip(range(1,k+1),colors):
            plt.fill_between([l1[i-1],r1[i-1]],[i,i], step='post', alpha=0.2, color=color)
            # plt.plot([l1[i-1],r1[i-1]],[i,i])
        plt.scatter(r,np.arange(1,k+1),color='k')
        plt.show()

r=np.array([1,2.125,2.875])
alpha=0.05
N=4
k=3
F_value=stats.f.ppf(1-alpha,dfn=k-1,dfd=(k-1)*(N-1))
TF=TF(calcTX2(N,k,r),N,k)
# print(stats.chi2.ppf(1-alpha,k-1))

print('F临界值:',F_value)
print('原始Friedman检验值为：',calcTX2(N,k,r))
print('新Friedman检验值为：',TF)
Nemenyi(TF,F_value,k)





































# print(stats.friedmanchisquare([1,1,1,1],[2,2.5,2,2],[3,2.5,3,3]))






