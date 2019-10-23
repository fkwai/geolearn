from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# training
tLst = [[20150501, 20151001], [20150402, 20160401]]
tagLst = ['2015RK', '2015']
for k in range(len(tLst)):
    optData = default.update(
        default.optDataSMAP,
        varT=['APCP_FORA'],
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=tLst[k],
        daObs=1)
    optModel = default.optLstmClose
    optLoss = default.optLossRMSE
    optTrain = default.update(default.optTrainSMAP, nEpoch=500)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA_Prcp_' + tagLst[k])
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    master.runTrain(masterDict, cudaID=(k+1) % 3, screen='DA' + tagLst[k])

    optData = default.update(
        default.optDataSMAP,
        varT=['APCP_FORA'],
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=tLst[k])
    optModel = default.optLstm
    optLoss = default.optLossRMSE
    optTrain = default.update(default.optTrainSMAP, nEpoch=500)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                       'CONUSv2f1_LSTM_Prcp_'+tagLst[k])
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    master.runTrain(masterDict, cudaID=(k+1) % 3, screen='LSTM' + tagLst[k])

# training
tLst = [[20150501, 20151001]]
yrLst = ['2015RK']
for k in range(len(tLst)):
    optData = default.update(
        default.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=tLst[k],
        daObs=1)
    optModel = default.optLstmClose
    optLoss = default.optLossRMSE
    optTrain = default.update(default.optTrainSMAP, nEpoch=500)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA_' + yrLst[k])
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    master.runTrain(masterDict, cudaID=2, screen='DA' + yrLst[k])

    # optData = default.update(
    #     default.optDataSMAP,
    #     rootDB=pathSMAP['DB_L3_NA'],
    #     subset='CONUSv2f1',
    #     tRange=tLst[k])
    # optModel = default.optLstm
    # optLoss = default.optLossRMSE
    # optTrain = default.update(default.optTrainSMAP, nEpoch=300)
    # out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM'+yrLst[k])
    # masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    # master.runTrain(masterDict, cudaID=k % 3, screen='LSTM' + yrLst[k])
