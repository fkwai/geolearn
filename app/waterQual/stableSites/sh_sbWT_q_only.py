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
subsetLst = ['{}-Y{}'.format('comb0', x) for x in [1, 2]]
subset = subsetLst[0]
saveName = '{}-{}-{}-{}'.format(dataName, 'streamflow_Y1')
varX = gridMET.varLst
varY = ['00060']
varYC = None
caseName = basins.wrapMaster(
    dataName=dataName, trainName=subset, batchSize=[None, 100],
    outName=saveName, varX=varX, varY=varY, varYC=varYC)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
