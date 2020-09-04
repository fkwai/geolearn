from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataName = 'ssWT'
wqData = waterQuality.DataModelWQ(dataName)
code = '00940'
ep = 500
label = 'ntnS'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
master = basins.loadMaster(outName)

weekly = True
siteNoLst = wqData.subsetInfo(trainSet)['siteNo'].unique().tolist()
sd = np.datetime64('1979-01-01')
ed = np.datetime64('2020-01-01')
retest = True

# run sequence test for all sites, default to be from first date to last date
if type(siteNoLst) is not list:
    siteNoLst = [siteNoLst]
master = basins.loadMaster(outName)
if ep is None:
    ep = master['nEpoch']
outDir = basins.nameFolder(outName)
sdS = pd.to_datetime(sd).strftime('%Y%m%d')
edS = pd.to_datetime(ed).strftime('%Y%m%d')
if weekly is True:
    saveDir = os.path.join(outDir, 'seqW-{}-{}-ep{}'.format(sdS, edS, ep))
else:
    saveDir = os.path.join(outDir, 'seq-{}-{}-ep{}'.format(sdS, edS, ep))
if not os.path.exists(saveDir):
    os.mkdir(saveDir)
siteSaveLst = os.listdir(saveDir)
if retest is True:
    sitePredLst = siteNoLst
else:
    sitePredLst = [
        siteNo for siteNo in siteNoLst if siteNo not in siteSaveLst]

# if len(sitePredLst) != 0:
#     if wqData is None:
#         wqData = waterQuality.DataModelWQ(master['dataName'])

(varX, varXC, varY, varYC) = (
    master['varX'], master['varXC'], master['varY'], master['varYC'])
(statX, statXC, statY, statYC) = basins.loadStat(outName)
model = basins.loadModel(outName, ep=ep)
tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)
# for siteNo in sitePredLst:

  siteNo = sitePredLst[0]
   if 'DRAIN_SQKM' in varXC:
        area = tabG[tabG.index == siteNo]['DRAIN_SQKM'].values[0]
    else:
        area = None
    # test model
    print('testing {} from {} to {}'.format(siteNo, sdS, edS))
    dfX = waterQuality.readSiteX(
        siteNo, varX, sd=sd, ed=ed, area=area, nFill=5)
    xA = np.expand_dims(dfX.values, axis=1)
    xcA = np.expand_dims(
        tabG.loc[siteNo].values.astype(np.float), axis=0)
    mtdX = wqData.extractVarMtd(varX)
    x = transform.transInAll(xA, mtdX, statLst=statX)
    mtdXC = wqData.extractVarMtd(varXC)
    xc = transform.transInAll(xcA, mtdXC, statLst=statXC)
    [x, xc] = trainTS.dealNaN([x, xc], master['optNaN'][:2])
    yOut = trainTS.testModel(model, x, xc)
    # transfer out
    nt = len(dfX)
    ny = len(varY) if varY is not None else 0
    nyc = len(varYC) if varYC is not None else 0
    yP = np.full([nt, ny+nyc], np.nan)
    yP[:, :ny] = wqData.transOut(yOut[:, 0, :ny], statY, varY)
    yP[:, ny:] = wqData.transOut(yOut[:, 0, ny:], statYC, varYC)
    # save output
    t = dfX.index.values.astype('datetime64[D]')
    colY = [] if varY is None else varY
    colYC = [] if varYC is None else varYC
    dfOut = pd.DataFrame(data=yP, columns=[colY+colYC], index=t)
    dfOut.index.name = 'date'
    dfOut = dfOut.reset_index()
    dfOut.to_csv(os.path.join(saveDir, siteNo), index=False)

freq='D'
area=None
nFill=5
sd=np.datetime64('1979-01-01')
ed=np.datetime64('2020-01-01')

if freq='D':
    tr = pd.date_range(sd, ed)
elif freq='W':
    tr = pd.date_range(sd,ed, freq='W-TUE')


t = pd.date_range(sd,ed, freq='W-TUE')


dfX = pd.DataFrame({'date': tr}).set_index('date')
# extract data
dfF = gridMET.readBasin(siteNo)
if '00060' in varX or 'runoff' in varX:
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    if 'runoff' in varX:
        if area is None:
            tabArea = gageII.readData(
                varLst=['DRAIN_SQKM'], siteNoLst=[siteNo])
            area = tabArea['DRAIN_SQKM'].values[0]
        dfQ['runoff'] = calRunoffArea(dfQ['00060'], area)
    dfX = dfX.join(dfQ)
dfX = dfX.join(dfF)
dfX = dfX[varX]
dfX = dfX.interpolate(limit=nFill, limit_direction='both')
