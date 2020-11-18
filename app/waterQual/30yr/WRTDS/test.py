import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from hydroDL import utils

n = 10000
np.random.seed(28041990)
a = np.random.normal(20, 0.1, size=n)
scipy.stats.kstest(a/np.std(a), 'norm')
scipy.stats.anderson(a, dist='norm')
scipy.stats.chisquare(a)
scipy.stats.jarque_bera(a)

scipy.stats.shapiro(a)

fig, ax = plt.subplots(1, 1)
ax.hist(a, bins=100)
fig.show()


distName = 'cauchy'
fig, ax = plt.subplots(1, 1)
dist = getattr(scipy.stats, distName)
data = dist(20, 50).rvs(size=1000)
x2 = utils.rmExt(data, p=5)

args = dist.fit(x2)
s, p = scipy.stats.kstest(x2, distName, args=args)
xx = np.sort(x2)
yy = dist(*args).pdf(xx)
_ = ax.hist(x2, bins=100, density=True)
_ = ax.plot(xx, yy, '-r')
fig.show()
