from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib


# ntn variables
dataName = 'rbWN5'
# wqData = waterQuality.DataModelWQ(dataName)

varF = gridMET.varLst
varQ = ['runoff']
varP = ntn.varLst
varT = ['sinT', 'cosT']

code = 'comb'
label = 'QFP_C'
varX = varQ+varF+varP
varY = None
varYC = usgs.newC
codeLst = ['00095', '00915', '00945', '00618']
dataNameLst = ['rbWN5-S{}'.format(code) for code in codeLst]
caseLst = list()
for dataName in dataNameLst:
    trainSet = 'comb-B10'
    saveName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=trainSet, batchSize=[None, 500],
        outName=saveName, varX=varX, varY=varY, varYC=varYC)
    caseLst.append(caseName)

# cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
# for caseName in caseLst:
#     slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)

basins.trainModelTS(caseName)
