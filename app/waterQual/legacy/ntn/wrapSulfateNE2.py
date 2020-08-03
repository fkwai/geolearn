import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.app import waterQuality

varC = usgs.varC
siteNoLst = ['0143400680', '01434021', '01434025']
nFill = 5
varG = gageII.lstWaterQuality
caseName = 'sulfateNE-daily'
rho = 365


# temp: read NTN
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)
tab = tabData[tabData['siteID'] == 'NY68']
varPLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
dfP = pd.DataFrame(columns=varPLst)
for k in range(len(tab)):
    t1 = pd.to_datetime(tab.iloc[k]['dateon']).date()
    t2 = pd.to_datetime(tab.iloc[k]['dateoff']).date()
    tt = pd.date_range(t1, t2)[:-1]
    data = np.tile(tab.iloc[k][varPLst].values, [len(tt), 1])
    tabTemp = pd.DataFrame(index=tt, columns=varPLst, data=data)
    dfP = dfP.append(tabTemp)
dfP.dropna(how='all')

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)

# gageII
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

# read data and merge to: f/q=[nT,nP,nX], g/c=[nP,nY]
fLst = list()  # forcing ts
gLst = list()  # geo-const
qLst = list()  # streamflow
cLst = list()  # water quality
cfLst = list()  # water quality flags
infoLst = list()
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    t1 = time.time()
    dfC, dfCF = usgs.readSample(
        siteNo, codeLst=varC, startDate=startDate, flag=2)
    dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfF = gridMET.readBasin(siteNo)
    for k in range(len(dfC)):
        ct = dfC.index[k]
        ctR = pd.date_range(ct-pd.Timedelta(days=rho-1), ct)
        if (ctR[0] < startDate) or (ctR[-1] > endDate):
            continue
        tempQ = pd.DataFrame({'date': ctR}).set_index('date').join(
            dfQ).interpolate(limit=nFill, limit_direction='both')
        tempF = pd.DataFrame({'date': ctR}).set_index('date').join(
            dfF).join(dfP).interpolate(limit=nFill, limit_direction='both')
        qLst.append(tempQ.values)
        fLst.append(tempF.values)
        cLst.append(dfC.iloc[k].values)
        cfLst.append(dfCF.iloc[k].values)
        gLst.append(tabG.loc[siteNo].values)
        infoLst.append(dict(siteNo=siteNo, date=ct))
    t2 = time.time()
    print('{} on site {} reading {:.3f} total {:.3f}'.format(
        i, siteNo, t2-t1, t2-t0))
q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)
c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float32)
cf = np.stack(cfLst, axis=-1).swapaxes(0, 1).astype(np.float32)
infoDf = pd.DataFrame(infoLst)
# add runoff
runoff = waterQuality.calRunoff(q[:, :, 0], infoDf)
q = np.stack([q[:, :, 0], runoff], axis=-1).astype(np.float32)
saveFolder = os.path.join(kPath.dirWQ, 'trainData')
saveName = os.path.join(saveFolder, caseName)
np.savez(saveName, q=q, f=f, c=c, g=g, cf=cf)
infoDf.to_csv(saveName+'.csv')
dictData = dict(name=caseName, rho=rho, nFill=nFill,
                varG=varG, varC=varC, varQ=['00060', 'runoff'],
                varF=gridMET.varLst+varPLst, siteNoLst=siteNoLst)
with open(saveName+'.json', 'w') as fp:
    json.dump(dictData, fp, indent=4)
