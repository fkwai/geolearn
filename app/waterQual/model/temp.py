from hydroDL import kPath
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.data import gridMET, usgs, gageII
import json

import os
import pandas as pd
import numpy as np

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]


wqData = waterQuality.DataModelWQ('HBN')

varX = usgs.varQ+gridMET.varLst


varF = gridMET.varLst+['00060']
varG = wqData.varG
varQ = ['00060']
varC = wqData.varC[:3]
varTup = (varF, varG, varQ, varC)
dataTup, statTup = wqData.transIn(
    subset=None, varTup=varTup)


defaultMaster = dict(
    dataName='HBN', trainName='first50', outName=None, modelName='CudnnLSTM',
    hiddenSize=256, batchSize=[None, 500], nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    varX=gridMET.varLst, varXC=gageII.lstWaterQuality,
    varY=usgs.varQ, varYC=usgs.varC
)


def wrapMaster(**kw):
    # default parameters
    dictPar = defaultMaster.copy()
    dictPar.update(kw)
    return dictPar


dictPar = wrapMaster(b=5,c=66,varXC=100)

dictPar.keys()
diff = list(set(dictPar) - set(defaultMaster))
'adffaff '+' '.join(diff)