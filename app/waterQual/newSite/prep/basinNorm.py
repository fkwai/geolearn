from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import json

codeLst = sorted(usgs.newC)
# dataName = 'nbWT'
dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.info.siteNo.unique()
info = wqData.info

c1Mat = np.ndarray(wqData.c.shape)
c2Mat = np.ndarray(wqData.c.shape)
for code in wqData.varC:
    ic = wqData.varC.index(code)
    for siteNo in siteNoLst:
        indS = info[info['siteNo'] == siteNo].index.values
        data = wqData.c[indS, ic]
        if len(indS) > 0:
            c1Mat[indS, ic] = np.nanpercentile(data, 10)
            c2Mat[indS, ic] = np.nanpercentile(data, 90)
        else:
            c1Mat[indS, ic] = 0
            c2Mat[indS, ic] = 1


c = (wqData.c-c1Mat)/(c2Mat-c1Mat)
q = wqData.q
f = wqData.f
g = wqData.g
varC = wqData.varC
varQ = wqData.varQ
varG = wqData.varG
varF = wqData.varF
infoDf = info
rho = wqData.rho
siteNoLst = wqData.siteNoLst
caseName = dataName+'_norm'

# # save
saveFolder = os.path.join(kPath.dirWQ, 'trainData')
saveName = os.path.join(saveFolder, caseName)
np.savez(saveName, q=q, f=f, c=c, g=g, c1=c1Mat, c2=c2Mat)
infoDf.to_csv(saveName+'.csv')
dictData = dict(name=caseName, rho=rho,
                varG=varG, varC=varC, varQ=varQ,
                varF=varF, siteNoLst=siteNoLst)
with open(saveName+'.json', 'w') as fp:
    json.dump(dictData, fp, indent=4)
