import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality as wq
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
if False:
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, codeLst)
    corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)
    # load basin attributes
    dfG = gageII.readData(siteNoLst=siteNoLst)
    dfG = gageII.updateRegion(dfG)
    dfG = gageII.updateCode(dfG)

indC = [codeLst.index(code) for code in codeLst2]
labelLst = ['{} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
            for code in codeLst2]
xMat = corrMat[:, indC, 1]
yMat = corrMat[:, indC, 2]
nXY = [nfx, nfy]