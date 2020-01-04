
import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import imp
import matplotlib
imp.reload(rnnSMAP)
rnnSMAP.reload()

doOpt = []
doOpt.append('train')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015, varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu',
        model='cudnn', loss='sigma'
    )
    for kk in range(1, 2):
        for k in range(12):
            trainName = 'ecoComb{}_v2f1'.format(k)
            opt['train'] = trainName
            if kk == 0:
                opt['var'] = 'varLst_soilM'
                opt['out'] = trainName+'_y15_soilM'
            elif kk == 1:
                opt['var'] = 'varLst_Forcing'
                opt['out'] = trainName+'_y15_Forcing'
            cudaID = k % 3
            print(trainName)
            runTrainLSTM.runCmdLine(
                opt=opt, cudaID=cudaID, screenName=opt['out'])
