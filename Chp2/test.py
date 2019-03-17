import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_data(func):
    def wrap(*args):
        k,n,p= args
        data = func(k,n,p)
        print('Binomial data:\n%s' % data)
        plt.plot(k, data, 'o-')
        plt.title('Binomial:n=%i,p=%.2f' % (n, p), fontsize=15)
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.show()
        # return func(*args)
    return wrap

@plot_data
def creat_binomial_data(k=np.arange(100),n=100,p=0.3):
    data = stats.binom.pmf(k, n, p)
    return data
# creat_binomial_data(np.arange(11),10,0.3)

def test_binomial(alpha=0.05,n=10,epsilon0=0.3):
    print(stats.binom.ppf(1-alpha/2,n,epsilon0)/n)
test_binomial()



#生成符合二项分布的数据
# print(np.random.binomial(10,0.3,size=100))

# binom_sim=stats.binom.rvs(n=10,p=0.3,size=20000)
# bins = np.arange(-0.5,10.5,1)
# ax=plt.subplot(111)
# plt.hist(binom_sim,bins=bins,density=True)
# ax.set_xticks(np.arange(12))
# plt.show()