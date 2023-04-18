import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt


dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

dm = dbVeg.DataModelVeg(df, subsetName='all')

varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ['Fpar', 'Lai']

indTrain = df.loadSubset('5fold_0_train')
indTest = df.loadSubset('5fold_0_test')


wS = 6
wL = 8
wM = 2
np.max([wS, wL, wM])

xLst = list()
yLst = list()
for iSite in indTrain:
    [y, t], ind = utils.rmNan([df.y[:, iSite], df.t])
    for k, i in enumerate(ind):
        if i > np.max([wS, wL, wM]) / 2:
            iS = [df.varX.index(var) for var in varS]
            vS = np.nanmean(df.x[i - wS : i + wS, iSite, iS], axis=0)
            iL = [df.varX.index(var) for var in varL]
            vL = np.nanmean(df.x[i - wL : i + wL, iSite, iL], axis=0)
            iM = [df.varX.index(var) for var in varM]
            vM = np.nanmean(df.x[i - wM : i + wM, iSite, iM], axis=0)
            xLst.append(np.concatenate([vS, vL, vM, df.xc[iSite, :]]))
            yLst.append(y[k])
x = np.stack(xLst, axis=0)
y = np.stack(yLst, axis=0)

# linear regression
from sklearn import linear_model

regr = linear_model.LinearRegression()
# drop nan
b1 = np.isnan(x).any(axis=1)
xx = x[~b1, :]
yy = y[~b1].flatten()

regr.fit(xx, yy)
yP = regr.predict(xx)

# plot
fig, ax = plt.subplots(1, 1)
ax.plot(yy, yP, '*')
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
fig.show()

