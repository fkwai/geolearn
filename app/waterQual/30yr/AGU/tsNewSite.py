from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

siteNo = '01545600'
code = '00955'
dataName = 'nbW'
labelLst = ['QF_C', 'QFP_C']
trainSet = '{}-B16'.format('comb')

# WRTDS
varF = gridMET.varLst
varP = ntn.varLst[2:3]
varQ = '00060'
varLst = ['00060', '00955']+varF+varP
varX = varF+varP
df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')
dfX = pd.DataFrame({'date': df.index}).set_index('date')
sn = 1
dfX = dfX.join(np.log(df[varQ]+sn)).rename(
    columns={varQ: 'logQ'})
dfX = dfX.join(df[varP])
yr = dfX.index.year.values
t = yr+dfX.index.dayofyear.values/365
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)
ind = np.where(yr < 2010)[0]
dfYP = pd.DataFrame(index=df.index, columns=[code])
dfYP.index.name = 'date'
dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
# dfXN = dfX
x = dfXN.iloc[ind].values
# y = np.log(df.iloc[ind][code].values+sn)
y = df.iloc[ind][code].values
[xx, yy], iv = utils.rmNan([x, y])
if len(yy) > 0:
    # yy, ind = utils.rmExt(yv, p=2.5, returnInd=True)
    # xx = xv[ind, :]
    lrModel = LinearRegression()
    lrModel = lrModel.fit(xx, yy)
    b = dfXN.isna().any(axis=1)
    yp = lrModel.predict(dfXN[~b].values)
    dfYP.at[dfYP[~b].index, code] = yp


figP, axP = plt.subplots(1, 1, figsize=(8, 2.5))
outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QF_C', trainSet)
outName2 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QFP_C', trainSet)

dfL1 = basins.loadSeq(outName1, siteNo)
dfL2 = basins.loadSeq(outName2, siteNo)
dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
t = dfO.index
# ts
tBar = np.datetime64('2016-01-01')
sd = np.datetime64('1980-01-01')
legLst = ['LSTM w/o rainfall chem', 'LSTM w rainfall chem', 'Observation']
axplot.plotTS(axP, t, [dfL1[code], dfL2[code],  dfO[code]],
              tBar=tBar, sd=sd, styLst='--*', cLst='rbk', legLst=legLst)
axP.legend()
figP.show()


figP, axP = plt.subplots(1, 1, figsize=(8, 2.5))
outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QF_C', trainSet)
outName2 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QFP_C', trainSet)
dfL1 = basins.loadSeq(outName1, siteNo)
dfL2 = basins.loadSeq(outName2, siteNo)
dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
t = dfO.index
# ts
tBar = np.datetime64('2016-01-01')
sd = np.datetime64('1980-01-01')
legLst = ['WRTDS w rainfall chem', 'LSTM w rainfall chem', 'Observation']
axplot.plotTS(axP, t, [dfYP[code], dfL2[code],  dfO[code]],
              tBar=tBar, sd=sd, styLst='--*', cLst='rbk', legLst=legLst)
axP.legend()
figP.show()