from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib

# for a test on training to resolve warnings
varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']

# ntn variables
dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)
codeLst = usgs.newC + ['comb']
labelLst = ['QFP_C', 'FP_CQ']
varF = gridMET.varLst
varQ = ['00060']
varP = ntn.varLst
caseLst = list()
for code in codeLst:
    if code == 'comb':
        varC = usgs.newC
    else:
        varC = [code]
    for label in labelLst:
        if label == 'QFP_C':
            varX = varQ+varF+varP
            varY = None
            varYC = varC
        elif label == 'FP_CQ':
            varX = varF+varP
            varY = varQ
            varYC = varC
        trainSet = '{}-B16'.format(code)
        saveName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=trainSet, batchSize=[None, 100],
            outName=saveName, varX=varX, varY=varY, varYC=varYC)
        caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
