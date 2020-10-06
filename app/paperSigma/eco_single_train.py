import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import huc_single_test
import imp
import matplotlib
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test huc vs huc

doOpt = []
doOpt.append('train')
# doOpt.append('test')
# doOpt.append('loadData')
# doOpt.append('crdMap')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'ecoSingle')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015, varC='varConstLst_Noah',
        dr=0.6, modelOpt='relu',
        model='cudnn', loss='sigma'
    )    
    for k in range(0, 17):
        trainName = 'ecoRegion'+str(k+1).zfill(2)+'_v2f1'
        opt['train'] = trainName
        opt['var'] = 'varLst_Forcing'
        opt['out'] = trainName+'_y15_Forcing'
        cudaID = k % 3
        print(trainName)
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaID, screenName=opt['out'])
            