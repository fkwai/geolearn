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
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# comb model
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
codeLst = wqData.varC
# labelLst = ['QFP_C', 'QTFP_C', 'QT_C']
for label in labelLst:
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    basins.testModelSeq(outName, siteNoLst, wqData=wqData, retest=True)

# # solo models
# label = 'FP_QC'
# codeLst = usgs.newC
# for code in codeLst:
#     trainSet = '{}-B10'.format(code)
#     outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
#     basins.testModelSeq(outName, siteNoLst, wqData=wqData)
