from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib
import os
import json

varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictSB_0412.json')) as f:
    dictSite = json.load(f)

# ntn variables
dataName = 'sbWT'
outNameLst = list()
outCodeLst = list()
wqData = waterQuality.DataModelWQ(dataName)
codeLst = wqData.varC
for code in codeLst:
    labelLst = ['ntn', 'ntnq']
    for label in labelLst:
        subsetLst = ['{}-Y{}'.format(code, x) for x in [1, 2]]
        subset = subsetLst[0]
        outName = '{}-{}-{}-{}'.format(dataName, code, label, subset)
        siteNoLst = dictSite[code]
        basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# test for streamflow
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)
subsetLst = 'comb0-Y1'
saveName = '{}-{}'.format(dataName, 'streamflow-Y1')
siteNoLst = wqData.siteNoLst
basins.testModelSeq(saveName, siteNoLst, wqData=wqData)
