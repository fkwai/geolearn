from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
import torch
import os
import json
import numpy as np

# sE=50
# nE=100
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp1', optQ=1)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp2', optQ=2)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp3', optQ=3)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp4', optQ=4)

# load master
modelName = 'temp1'
modelFolder = os.path.join(kPath.dirWQ, 'model', modelName)
masterFile = os.path.join(modelFolder, 'master.json')
with open(masterFile, 'r') as fp:
    master = json.load(fp)
statFile = os.path.join(modelFolder, 'stat.json')
with open(statFile, 'r') as fp:
    dictStat = json.load(fp)
statX = dictStat['statX']
statXC = dictStat['statXC']
statY = dictStat['statY']
statYC = dictStat['statYC']


# load data
wqData = waterQuality.DataModelWQ(master['dataName'])
dataLst, statLst = wqData.transIn(subset='first80', optQ=master['optQ'])

# load model
nEpoch = 100
modelFile = os.path.join(modelFolder, 'model_ep{}'.format(nEpoch))
model = torch.load(modelFile)
sizeLst = trainTS.getSize(dataLst)
dataLst = trainTS.dealNaN(dataLst, master['optNan'])
x = dataLst[0]
xc = dataLst[1]
ny = sizeLst[2]

# np.where(np.isnan(dataLst[0]))

# test model - point by point
yOut, ycOut = trainTS.testModel(model, x, xc, ny)
yP, ycP = wqData.transOut(yOut, ycOut, statLst[2], statLst[3])

# plot
iS = 0
iC = 8
siteNoLst = wqData.info.siteNo.unique()
dfS = wqData.info[wqData.info['siteNo'] == siteNoLst[iS]]
ind = dfS.index.values
t = dfS.date.values.astype(np.datetime64)
obs = wqData.c[ind, iC]
pred = ycP[ind, iC]
import matplotlib.pyplot as plt
plt.plot(obs,pred,'*')
plt.show()
