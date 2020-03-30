from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# train
optData = default.update(
    default.optDataSMAP,
    rootDB=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150402, 20180401],
    daObs=1)
optModel = default.optLstmClose
optLoss = default.optLossRMSE
optTrain = default.update(default.optTrainSMAP, nEpoch=500)
out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUS_DA_3yr')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.runTrain(masterDict, cudaID=2, screen='DA')

# test
subset = 'CONUS'
tRange = [20150402, 20180401]
# out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA2015')
out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUS_DA_3yr')
df, yf, obs = master.test(out, tRange=tRange, subset=subset, batchSize=100)

# /home/kxf227/work/GitHUB/hydroDL-dev/app/closeLoop/fullCONUS.py