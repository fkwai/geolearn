
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import pandas as pd
import json
import os

dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
rho = wqData.rho
varG = wqData.varG
varC = wqData.varC
varQ = wqData.varQ
varF = wqData.varF
siteNoLst = wqData.siteNoLst
info = wqData.info
q = wqData.q.copy()
f = wqData.f.copy()
cs = wqData.c.copy()
g = wqData.g.copy()

for siteNo in siteNoLst:
    indS = info[info['siteNo'] == siteNo].index.values
    for k in range(len(varC)):
        x = cs[indS, k].copy()
        ind = np.where(~np.isnan(x))[0]
        ind2 = ind.copy()
        np.random.shuffle(ind2)
        cs[indS[ind], k] = x[ind2]

codeLst = ['00095', '00915', '00945', '00618']
for code in codeLst:
    caseName = dataName+'-S{}'.format(code)
    indC = varC.index(code)
    c = cs.copy()
    c[:, indC] = wqData.c[:, indC]

    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, q=q, f=f, c=c, g=g)
    info.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho,
                    varG=varG, varC=varC, varQ=varQ,
                    varF=varF, siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)
