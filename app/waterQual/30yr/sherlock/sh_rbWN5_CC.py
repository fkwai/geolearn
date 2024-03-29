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
dataName = 'rbWN5'
# wqData = waterQuality.DataModelWQ(dataName)
rmLst = [['00010'], ['00010', '00095'], ['00010', '00095', '00400']]
subsetLst = ['rmT', 'rmTK', 'rmTKH']
labelLst = ['QT_C', 'QTFP_C', 'QFP_C']

varF = gridMET.varLst
varQ = ['00060']
varP = ntn.varLst
varT = ['sinT', 'cosT']

caseLst = list()
for rmCode, subset in zip(rmLst, subsetLst):
    varXC = gageII.varLst+rmCode
    varYC = list(set(usgs.newC)-set(rmCode))
    for label in labelLst:
        if label == 'QFP_C':
            varX = varQ+varF+varP
            varY = None
        elif label == 'FP_QC':
            varX = varF+varP
            varY = varQ
        elif label == 'FP_Q':
            varX = varF+varP
            varY = varQ
            varYC = None
        elif label == 'F_QC':
            varX = varF
            varY = varQ
        elif label == 'QF_C':
            varX = varQ+varF
            varY = None
        elif label == 'FP_C':
            varX = varF+varP
            varY = None
        elif label == 'P_C':
            varX = varP
            varY = None
        elif label == 'Q_C':
            varX = varQ
            varY = None
        elif label == 'QT_C':
            varX = varQ+varT
            varY = None
        elif label == 'QTFP_C':
            varX = varQ+varT+varF+varP
            varY = None
        trainSet = '{}-B10'.format(subset)
        saveName = '{}-{}-{}-{}'.format(dataName, subset, label, trainSet)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=trainSet, batchSize=[None, 500],
            outName=saveName, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
        caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
