import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn, transform

varC = usgs.varC
nFill = 5
varG = gageII.lstWaterQuality
caseName = 'ssW'

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
with open(os.path.join(dirInv, 'dictStableSites_0610_0220.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# add a start/end date to improve efficiency.
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
sd = t[0]
ed = t[-1]
td = pd.date_range(sd, ed)
rho = 50

# gageII
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)
ntnFolder = os.path.join(kPath.dirData, 'EPA', 'NTN', 'usgs', 'weeklyRaw')
varPLst = ntn.varLst+['distNTN']

# read data and merge to: f/q=[nT,nP,nX], g/c=[nP,nY]
fLst = list()  # forcing ts
pLst = list()  # concentrations in rainfall
gLst = list()  # geo-const
qLst = list()  # streamflow
cLst = list()  # water quality
# cfLst = list()  # water quality flags

infoLst = list()
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    t1 = time.time()
    dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfF = gridMET.readBasin(siteNo)
    dfP = pd.read_csv(os.path.join(ntnFolder, siteNo), index_col='date')
    dfP = dfP[varPLst]

    # merge to one table
    df = pd.DataFrame({'date': td}).set_index('date')
    df = df.join(dfC)
    df = df.join(dfQ)
    df = df.join(dfF)
    df = df.rename(columns={'00060_00003': '00060'})

    # convert to weekly
    dfW = df.resample('W-TUE').mean()
    dfW = dfW.join(dfP)
    dfC = dfW[varC].dropna(how='all')
    for k in range(len(dfC)):
        ct = dfC.index[k]
        ctR = pd.date_range(
            start=ct-pd.Timedelta(days=rho*7-1), end=ct, freq='W-TUE')
        if (ctR[0] < sd) or (ctR[-1] > ed):
            continue
        tempQ = pd.DataFrame({'date': ctR}).set_index('date').join(
            dfW['00060']).interpolate(limit=nFill, limit_direction='both')
        tempF = pd.DataFrame({'date': ctR}).set_index('date').join(
            dfW[gridMET.varLst+varPLst]).interpolate(limit=nFill, limit_direction='both')
        qLst.append(tempQ.values)
        fLst.append(tempF.values)
        cLst.append(dfC.iloc[k].values)
        gLst.append(tabG.loc[siteNo].values)
        infoLst.append(dict(siteNo=siteNo, date=ct))
    t2 = time.time()
    print('{} on site {} reading {:.3f} total {:.3f}'.format(
        i, siteNo, t2-t1, t2-t0))
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float32)

# save
infoDf = pd.DataFrame(infoLst)
saveFolder = os.path.join(kPath.dirWQ, 'trainData')
saveName = os.path.join(saveFolder, caseName)
np.savez(saveName, q=q, f=f, c=c, g=g)
infoDf.to_csv(saveName+'.csv')
dictData = dict(name=caseName, rho=rho, nFill=nFill,
                varG=varG, varC=varC, varQ=['00060'],
                varF=gridMET.varLst+varPLst, siteNoLst=siteNoLst)
with open(saveName+'.json', 'w') as fp:
    json.dump(dictData, fp, indent=4)
