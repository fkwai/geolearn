from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath
import json
import os
import importlib
importlib.reload(axplot)
importlib.reload(figplot)


dm = dbBasin.DataModelFull('weathering')

# subset
dm.saveSubset('B10', ed='2009-12-31')
dm.saveSubset('A10', sd='2010-01-01')


yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(dm.t, yrIn, pick=False)
t2 = dbBasin.func.pickByYear(dm.t, yrIn)
dm.createSubset('pkYr5', dateLst=t1)
dm.createSubset('rmYr5', dateLst=t2)

codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
d1 = dbBasin.DataTrain(dm, varY=codeSel, subset='B10')
d2 = dbBasin.DataTrain(dm, varY=codeSel, subset='A10')
d1 = dbBasin.DataTrain(dm, varY=codeSel, subset='rmYr5')
d2 = dbBasin.DataTrain(dm, varY=codeSel, subset='pkYr5')


# check hist
varLst = codeSel
nd = len(varLst)
bins = 50
fig, axes = plt.subplots(nd, 2)
for k, var in enumerate(varLst):
    _ = axes[k, 0].hist(d1.y[:, :, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(d2.y[:, :, k].flatten(), bins=bins)
fig.show()

a1 = d1.Y.reshape(-1, d1.Y.shape[-1])
qt = QuantileTransformer(
    n_quantiles=50, random_state=0, output_distribution='normal')
# qt = PowerTransformer(method='yeo-johnson')

qt.fit(a1)
b1 = qt.transform(a1)
c1 = qt.inverse_transform(b1)
a2 = d2.Y.reshape(-1, d2.Y.shape[-1])
b2 = qt.transform(a2)
c2 = qt.inverse_transform(b2)

np.nansum(np.abs(a1-c1))
np.nansum(np.abs(a2-c2))

fig, axes = plt.subplots(nd, 2)
for k, var in enumerate(varLst):
    _ = axes[k, 0].hist(b1[:, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b2[:, k].flatten(), bins=bins)
    axes[k, 0].set_xlim([-3, 3])
    axes[k, 1].set_xlim([-3, 3])
fig.show()

fig, axes = plt.subplots(nd, 2)
for k, var in enumerate(varLst):
    _ = axes[k, 0].hist(c1[:, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(c2[:, k].flatten(), bins=bins)
fig.show()

A1 = a1.reshape(d1.Y.shape)
B1 = b1.reshape(d1.Y.shape)
C1 = c1.reshape(d1.Y.shape)
A2 = a2.reshape(d2.Y.shape)
B2 = b2.reshape(d2.Y.shape)
C2 = c2.reshape(d2.Y.shape)
# difference in and out
labelLst = list()
for ic, code in enumerate(codeSel):
    shortName = usgs.codePdf.loc[code]['shortName']
    temp = '{} {}'.format(
        code, shortName)
    labelLst.append(temp)

for k in range(len(dm.siteNoLst)):
    fig, axes = figplot.multiTS(
        d1.t, [C1[:, k, :], A1[:, k, :]], labelLst=labelLst, cLst='rk')
    fig.show()
np.nansum(np.abs(A1-C1))


for k in range(len(dm.siteNoLst)):
    fig, axes = figplot.multiTS(
        d2.t, [C2[:, k, :], A2[:, k, :]], labelLst=labelLst, cLst='rk')
    fig.show()
np.nansum(np.abs(A2-C2))


for k in range(len(dm.siteNoLst)):
    fig, axes = figplot.multiTS(
        d2.t, [B2[:, k, :]], labelLst=labelLst, cLst='rk')
    fig.show()
