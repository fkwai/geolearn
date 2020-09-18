from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib


varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']

# ntn variables
dataName = 'sbWT'
caseLst = list()
wqData = waterQuality.DataModelWQ(dataName)
codeLst = varNtnUsgsLst
label = 'ntnSq'
for code in codeLst:
    varX = ['00060'] + gridMET.varLst + \
        [varNtnLst[varNtnUsgsLst.index(code)], 'distNTN']
    varY = [code]
    varYC = None
    subsetLst = ['{}-Y{}'.format(code, x) for x in [1, 2]]
    # wrap up
    # for subset in subsetLst:
    subset = subsetLst[0]
    saveName = '{}-{}-{}-{}'.format(dataName, code, label, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 100],
        outName=saveName, varX=varX, varY=varY, varYC=varYC)
    caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
