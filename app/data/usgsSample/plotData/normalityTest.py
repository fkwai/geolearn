
from hydroDL.data import usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath
import os
from hydroDL.app.waterQuality import cqType
import importlib

import matplotlib.gridspec as gridspec

siteNo='01125100'
code='00915'
dfC,dfF=usgs.readSample(siteNo, codeLst=[code], flag=2)

# siteNo='01127000'
# code='00665'
# dfC,dfF=usgs.readSample(siteNo, codeLst=[code], flag=2,startDate='1979-01-01')
# df=dfC.dropna()
# fig, ax = plt.subplots(1, 1)
# ax.plot(df.index.values,df[code].values,'*')
# fig.show()

df=dfC.dropna()

fig, ax = plt.subplots(1, 1)
ax.plot(df.index.values,df[code].values,'*')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.hist(df[code].values,bins=20)
fig.show()

fig, ax = plt.subplots(1, 1)
ax.hist(np.log(df[code].values),bins=20)
fig.show()

import statsmodels.api as sm 
import pylab as py 
data = np.random.normal(0, 1, 100)     
  
data=df.values.flatten()


fig,ax= plt.subplots(1,1)
sm.qqplot(data, ax=ax,fit=True,label='norm',
          marker='.', markerfacecolor='k', markeredgecolor='k') 
sm.qqplot(np.log(data), ax=ax,fit=True,label='lognorm',
          marker='.', markerfacecolor='r', markeredgecolor='r') 
sm.qqline(ax, line='45', fmt='k--')
ax.legend()
fig.show()


import scipy
from scipy.stats import shapiro,kstest,kurtosis,skew
import random

kurtosis(data)
kurtosis(np.log(data))

skew(data)
skew(np.log(data))

data=df.values.flatten()

random.shuffle(data)


random.shuffle(data)
shapiro(data[:50])
shapiro(data[:100])
shapiro(data[:150])

nLst=[50,100,150,200,252]
data=df.values.flatten()
random.shuffle(data)
for n in nLst:
    res=shapiro(np.log(data[:n]))
    print('size {} p {}'.format(n,res[1]))


shapiro(np.log(data[:50]))
shapiro(np.log(data[:100]))
shapiro(np.log(data[:150]))
shapiro(np.log(data[:200]))
shapiro(np.log(data))



shapiro(data)
shapiro(np.log(data))

data=df.values.flatten()
xx=data[(data<np.percentile(data,90)) & (data>np.percentile(data,10))]
shapiro(xx)
shapiro(np.log(xx))


import statsmodels.api as sm

from scipy.stats import normaltest,anderson
data=df.values.flatten()
data=data[:150]
normaltest(data)
normaltest(np.log(data))
result=anderson(data)
result=anderson(np.log(data))
for i in range(len(result.critical_values)):
    slevel, cvalues = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (slevel, cvalues))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (slevel, cvalues))

anderson(np.log(data))


data=df.values.flatten()
sm.stats.lilliefors(data)
sm.stats.lilliefors(np.log(data))
sm.stats.normal_ad(data)
sm.stats.normal_ad((np.log(data)))

sm.stats.normal_ad((np.log(data[:100])))


x = np.random.normal(size=300)
shapiro(x)

scipy.stats.probplot(data)
