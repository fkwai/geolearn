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

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'weathering'
freq = 'D'
# DM = dbBasin.DataModelFull.new(
#     dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)
DM = dbBasin.DataModelFull(dataName)


# check hist
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']

varLst = codeSel
# varLst = usgs.varQ
a = DM.extractVarT(varLst)
mtd = dbBasin.io.extractVarMtd(varLst)
b, s = transform.transIn(a, mtd)
c = transform.transOut(b, mtd, s)

b = np.ndarray(a.shape)
qtLst = list()
for k, code in enumerate(codeSel):
    qt = QuantileTransformer(
        n_quantiles=10, random_state=0, output_distribution='normal')
    qt.fit(a[:, :, k])
    b[:, :, k] = qt.transform(a[:, :, k])
    c[:, :, k] = qt.inverse_transform(b[:, :, k])
    qtLst.append(qt)
np.nansum(np.abs(a-c))

nd = len(varLst)
bins = 20
fig, axes = plt.subplots(nd, 2)
for k, var in enumerate(varLst):
    _ = axes[k, 0].hist(a[..., k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b[..., k].flatten(), bins=bins)
fig.show()

# difference in and out
labelLst = list()
for ic, code in enumerate(codeSel):
    shortName = usgs.codePdf.loc[code]['shortName']
    temp = '{} {}'.format(
        code, shortName)
    labelLst.append(temp)
k = 0
for k in range(len(siteNoLst)):
    fig, axes = figplot.multiTS(
        DM.t, [c[:, k, :], a[:, k, :]], labelLst=labelLst, cLst='rk')
    fig.show()


pt = PowerTransformer(method='box-cox', standardize=False)
