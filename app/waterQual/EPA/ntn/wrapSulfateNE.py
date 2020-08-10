import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform

# varC = usgs.varC
varC = ['00945', '00935']
siteNoLst = ['0143400680', '01434021', '01434025']
nFill = 3
varG = gageII.lstWaterQuality
caseName = 'sulfateNE'

# add a start/end date to improve efficiency.
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
sd = t[0]
ed = t[-1]
td = pd.date_range(sd, ed)
rho = 50


# temp: read NTN
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)
tab = tabData[tabData['siteID'] == 'NY68']
tab.index = pd.to_datetime(tab['dateon'])
weekday = tab.index.normalize().weekday
tab2 = pd.DataFrame(index=t)
tol = pd.Timedelta(3, 'D')
tab2 = pd.merge_asof(left=tab2, right=tab, right_index=True,
                     left_index=True, direction='nearest', tolerance=tol)
varPLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
dfP = tab2[varPLst]


# gageII
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

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
    # merge to one table
    df = pd.DataFrame({'date': td}).set_index('date')
    df = df.join(dfC)
    df = df.join(dfQ)
    df = df.join(dfF)
    df = df.rename(columns={'00060_00003': '00060'})

    # convert to weekly
    offset = pd.offsets.timedelta(days=-6)
    dfW = df.resample('W-MON', loffset=offset).mean()
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
