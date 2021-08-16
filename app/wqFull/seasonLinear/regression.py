import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm

dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst


siteNo = siteNoLst[0]
# print('{}/{} {:.2f}'.format(
#     kk, len(siteNoLst), time.time()-t0))
# prep data
# saveName = os.path.join(dirOut, siteNo)
varQ = '00060'
varLst = DF.varC+[varQ]
df = dbBasin.io.readSiteTS(siteNo, varLst=varLst, freq='D')

dfX = pd.DataFrame({'date': df.index}).set_index('date')
yr = dfX.index.year.values
t = yr+dfX.index.dayofyear.values/365
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)
dfYP = pd.DataFrame(index=df.index, columns=codeLst)
dfYP.index.name = 'date'
# dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
dfXN = dfX
for code in codeLst:
x = dfXN.values
# y = np.log(df.iloc[ind][code].values+sn)
y = df[code].values
[xx, yy], iv = utils.rmNan([x, y])
if len(xx) > 0:
lrModel = LinearRegression()
lrModel = lrModel.fit(xx, yy)
b = dfXN.isna().any(axis=1)
yp = lrModel.predict(dfXN[~b].values)
# yp = np.exp(yp)-sn
dfYP.at[dfYP[~b].index, code] = yp
coef = lrModel.coef_
inte = lrModel.intercept_
parLst = [len(yy), coef[0], coef[1],  inte]
dictPar[code].loc[siteNo] = parLst]
