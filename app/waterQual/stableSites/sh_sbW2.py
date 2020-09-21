from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib

# ntn variables
dataName = 'sbW'
wqData = waterQuality.DataModelWQ(dataName)
codeLst = wqData.varC + ['comb0', 'comb1', 'comb2']
caseLst = list()
for code in codeLst:
    if code[:4] == 'comb':
        varC = wqData.varC
    else:
        varC = [code]
    label = 'ntnq'
    varX = ['00060']+gridMET.varLst+ntn.varLst
    varY = None
    subsetLst = ['{}-Y{}'.format(code, x) for x in [1, 2]]
    # wrap up
    # for subset in subsetLst:
    subset = subsetLst[0]
    saveName = '{}-{}-{}-{}'.format(dataName, code, label, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 100],
        outName=saveName, varX=varX, varY=varY, varYC=varC)
    caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
