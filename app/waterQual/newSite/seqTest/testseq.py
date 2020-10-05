from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib
import os
import json

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# ntn variables
dataName = 'nbW'
outNameLst = list()
outCodeLst = list()
wqData = waterQuality.DataModelWQ(dataName)
codeLst = wqData.varC
# labelLst = ['QFP_C', 'FP_QC']
# labelLst = ['F_QC', 'QF_C', 'FP_C', 'P_C']
labelLst = ['QT_C']

for label in labelLst:
    trainSet = 'comb-B16'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    basins.testModelSeq(outName, siteNoLst, wqData=wqData)
