import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

doLst = list()
# doLst.append('data')
# doLst.append('subset')
# doLst.append('train')

if 'data' in doLst:
    # only look at 5 site with most 00955 obs
    # ['11264500', '07083000', '01466500', '04063700', '10343500']
    dataName = 'HBN'
    codeLst = ['00618', '00955']
    wqData = waterQuality.DataModelWQ(dataName)
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    siteNoHBN = wqData.info['siteNo'].unique()
    info = wqData.info.iloc[indAll]
    tabCount = info.groupby('siteNo').count()
    siteNoLst = tabCount.nlargest(5, 'date').index.tolist()
    wqData = waterQuality.DataModelWQ.new('HBN5', siteNoLst)

if 'subset' in doLst:
    wqData = waterQuality.DataModelWQ('HBN5')
    codeLst = ['00618', '00955']
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    indAny = np.where(~np.isnan(wqData.c[:, icLst]).any(axis=1))[0]
    wqData.saveSubset('-'.join(sorted(codeLst)+['all']), indAll)
    wqData.saveSubset('-'.join(sorted(codeLst)+['any']), indAny)
    for ind, lab in zip([indAll, indAny], ['all', 'any']):
        indYr1 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[1979, 2000])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y8090']), indYr1)
        indYr2 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[2000, 2020])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y0010']), indYr2)

if 'training' in doLst:
    dataName = 'HBN5'
    codeLst = ['00618', '00955']
    trainset = '00618-00955-all-Y8090'
    testset = '00618-00955-all-Y0010'
    out = 'HBN5-00618-00955-all-Y8090'
    wqData = waterQuality.DataModelWQ(dataName)
    masterName = basins.wrapMaster(
        dataName='HBN5', trainName=trainset, batchSize=[
            None, 100], outName=out, varYC=codeLst, nEpoch=100)
    basins.trainModelTS(masterName)


# sequence testing
dataName = 'HBN'
outName = 'HBN-00618-00955-all-Y8090-opt2'
testset = '00618-00955-all'
wqData = waterQuality.DataModelWQ(dataName)

# # point testing
yP, ycP = basins.testModel(outName, testset, wqData=wqData)

dictP = basins.loadMaster(outName)
statTup = basins.loadStat(outName)
model = basins.loadModel(outName)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
siteNoLst = wqData.info['siteNo'].unique().tolist()

indTest = wqData.subset[testset]
infoTest = wqData.info.iloc[indTest].reset_index()

(varX, varXC, varY, varYC) = (
    dictP['varX'], dictP['varXC'], dictP['varY'], dictP['varYC'])
statX, statXC, statY, statYC = statTup
nFill = 5
rho = 365

# sequence test
tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

siteNo = siteNoLst[0]

# testset - only get sd ed
tTest = infoTest[infoTest['siteNo'] == siteNo]['date'].values
sdX = tTest[0]-np.timedelta64(rho-1, 'D')
sdY = tTest[0]
ed = tTest[-1]
trX = pd.date_range(sdX, ed)
trY = pd.date_range(sdY, ed)
dfX = pd.DataFrame({'date': trX}).set_index('date')
dfY = pd.DataFrame({'date': trY}).set_index('date')

# extract data
dfC = usgs.readSample(siteNo, codeLst=varYC, startDate=sdX)
dfF = gridMET.readBasin(siteNo)
dfQ = usgs.readStreamflow(siteNo, startDate=sdX)
dfQ = dfQ.rename(columns={'00060_00003': '00060'})
area = tabG[tabG.index == siteNo]['DRAIN_SQKM'].values
unitConv = 0.3048**3*365*24*60*60/1000**2
dfQ['runoff'] = dfQ['00060']/area*unitConv
if '00060' in varX or 'runoff' in varX:
    dfX = dfX.join(dfQ)
elif '00060' in varY or 'runoff' in varY:
    dfY = dfY.join(dfQ)
dfX = dfX.join(dfF)
dfY = dfY.join(dfC)
dfX = dfX[varX]
dfY = dfY[varY+varYC]

# normalize concat input data
dfX = dfX.interpolate(limit=nFill, limit_direction='both')
xA = np.expand_dims(dfX.values, axis=1)
xcA = np.expand_dims(tabG.loc[siteNo].values.astype(np.float), axis=0)
mtdX = wqData.extractVarMtd(varX)
x = transform.transInAll(xA, mtdX, statLst=statX)
mtdXC = wqData.extractVarMtd(varXC)
xc = transform.transInAll(xcA, mtdXC, statLst=statXC)

yP = trainTS.testModel(model, x, xc)

# # test
# nt = len(dfX)
# x, xc = trainTS.dealNaN((x, xc), dictP['optNaN'][:2])
# xx = np.concatenate([x, np.tile(xc[0, :], [1, nt, 1])], axis=-1).swapaxes(0, 1)
# xT = torch.from_numpy(xx).float()
# if torch.cuda.is_available():
#     xT = xT.cuda()
# # if i == 0 and ind1 == 0:
# #     try:
# #         yT = model(xT)
# #     except:
# #         print('first iteration failed again')
# yT = model(xT)
# yP = yT.detach().cpu().numpy()[:, 0, :]

nt = len(dfX)
ny = len(varY) if varY is not None else 0
nyc = len(varYC) if varYC is not None else 0
out = np.full([nt, ny+nyc], np.nan)
out[:, :ny] = wqData.transOut(yP[:, 0, :ny], statTup[2], varY)
out[:, ny:] = wqData.transOut(yP[:, 0, ny:], statTup[3], varYC)
pred = out[(rho-1):, :]
obs = dfY.values
t = dfY.index.values.astype('datetime64[D]')

# plot
indP = infoTest[infoTest['siteNo'] == siteNo].index.values
tP = infoTest[infoTest['siteNo'] == siteNo]['date'].values
fig, axes = plt.subplots(3, 1)
axplot.plotTS(axes[0], t, [pred[:, 0], obs[:, 0]],
              legLst=['pred', 'obs'], styLst='--')
for k in range(1, 3):
    axplot.plotTS(axes[k], t, [pred[:, k], obs[:, k]],
                  legLst=['pred', 'obs'], styLst='-*')
    # axplot.plotTS(axes[k], tP, [predP[indP, k-1], obsP[indP, k-1]],
    #               legLst=['predP', 'obsP'], styLst='**', cLst='mg')
fig.show()


importlib.reload(trainTS)
